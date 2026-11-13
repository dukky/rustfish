// CUDA kernels of nnue-trainer-cuda (compiled to PTX by build.rs).
//
// Deterministic: no float atomics; every sum has a fixed order (it differs
// from the CPU trainer's order, so nets are close to, not equal to, CPU
// nets). Layouts are the host's (src/main.rs): FT rows `[row][H]`, `rest`
// = FT bias (H) | output weights (B * 2H) | output bias (B).

#include <stdint.h>
#include <cuda_fp16.h>

#define INPUTS 768
#define CLIP 1.98f
// --head (docs/nnue-deeper-head.md): per output bucket W1 [H1][2H], b1,
// W2 [H2][H1], b2, W3 [H2], b3 after the output biases in `rest`; per
// sample scratch hs[HS] = z1 (H1) | z2 (H2) | dz1 (H1) | dz2 (H2), the CPU
// trainer's layout (nnue_trainer::HS). UNTESTED ON CUDA (written without a
// GPU; see README "Head: validating on the GPU").
#define H1 16
#define H2 32
#define HS 96
#define HSMALL (H1 + H2 * H1 + H2 + H2 + 1)
__device__ __forceinline__ int head_len(int H) { return H1 * 2 * H + HSMALL; }
__device__ __forceinline__ const float *head_of(const float *rest, int H, int buckets, int b) {
    return rest + H + buckets * 2 * H + buckets + (size_t)b * head_len(H);
}
__device__ __forceinline__ float sq01(float z) {
    float c = fminf(fmaxf(z, 0.f), 1.f);
    return c * c;
}
__device__ __forceinline__ float dsq01(float z) { return (z > 0.f && z < 1.f) ? 2.0f * z : 0.0f; }

struct GSample {
    uint16_t f[2][32];
    uint8_t n, bucket;
    uint16_t pad;
    float target;
};

struct Item {
    int bucket, start, end, range;
};

struct AdamC {
    float c1, c2, lr, decay, inv_batch;
};

// ---------------------------------------------------------------- forward

// One block per sample (visited in `order`), H/2 threads: thread t owns
// side t / (H/4) and four columns. Accumulators, output gradient, loss.
__device__ __forceinline__ float4 row4(const float *wt, size_t at) { return __ldg((const float4 *)(wt + at)); }
__device__ __forceinline__ float4 row4(const __half *wt, size_t at) {
    uint2 u = __ldg((const uint2 *)(wt + at));
    float2 a = __half22float2(*(const __half2 *)&u.x), b = __half22float2(*(const __half2 *)&u.y);
    return make_float4(a.x, a.y, b.x, b.y);
}

template <typename W>
__device__ __forceinline__ void forward_impl(const GSample *ss, const uint32_t *order, const W *wt, const float *rest,
                                             float *acc, float *gout, float *loss, int H, int buckets, float *hs,
                                             int hd) {
    int sample = order[blockIdx.x];
    const GSample &s = ss[sample];
    int q = H / 4, t = threadIdx.x, side = t / q, c = (t % q) * 4;
    const float *ftb = rest, *ow = rest + H, *ob = rest + H + buckets * 2 * H;
    // The sample's feature indices once into shared memory (a global index
    // load before every row load made the loop latency-bound).
    __shared__ uint32_t fs[32];
    __shared__ int ns;
    if (t < 32) fs[t] = ((const uint32_t *)s.f)[t];
    if (t == 0) ns = s.n;
    __syncthreads();
    const uint16_t *feats = (const uint16_t *)fs + side * 32;
    int n = ns;
    float4 a = *(const float4 *)(ftb + c);
    // Four row loads in flight, added in feature order.
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        float4 w0 = row4(wt, (size_t)feats[i] * H + c);
        float4 w1 = row4(wt, (size_t)feats[i + 1] * H + c);
        float4 w2 = row4(wt, (size_t)feats[i + 2] * H + c);
        float4 w3 = row4(wt, (size_t)feats[i + 3] * H + c);
        a.x += w0.x; a.y += w0.y; a.z += w0.z; a.w += w0.w;
        a.x += w1.x; a.y += w1.y; a.z += w1.z; a.w += w1.w;
        a.x += w2.x; a.y += w2.y; a.z += w2.z; a.w += w2.w;
        a.x += w3.x; a.y += w3.y; a.z += w3.z; a.w += w3.w;
    }
    for (; i < n; i++) {
        float4 w = row4(wt, (size_t)feats[i] * H + c);
        a.x += w.x; a.y += w.y; a.z += w.z; a.w += w.w;
    }
    *(float4 *)(acc + ((size_t)sample * 2 + side) * H + c) = a;
    const float *w = ow + s.bucket * 2 * H + side * H + c;
    float x0 = fminf(fmaxf(a.x, 0.f), 1.f), x1 = fminf(fmaxf(a.y, 0.f), 1.f), x2 = fminf(fmaxf(a.z, 0.f), 1.f),
          x3 = fminf(fmaxf(a.w, 0.f), 1.f);
    float part = x0 * x0 * w[0] + x1 * x1 * w[1] + x2 * x2 * w[2] + x3 * x3 * w[3];
    for (int o = 16; o; o >>= 1) part += __shfl_down_sync(0xffffffff, part, o);
    __shared__ float red[32];
    if ((t & 31) == 0) red[t >> 5] = part;
    if (hd) {
        // Head L1 partials of this thread's 4 columns, per z1 row q, then a
        // fixed-order reduction: warp shuffle tree, then warps in order.
        __shared__ float hred[32][H1];
        __shared__ float z1s[H1], dz2s[H2];
        const float *hp = head_of(rest, H, buckets, s.bucket);
        const float *w1 = hp + side * H + c;
        float a0 = x0 * x0, a1 = x1 * x1, a2 = x2 * x2, a3 = x3 * x3;
#pragma unroll
        for (int q = 0; q < H1; q++) {
            const float *r = w1 + (size_t)q * 2 * H;
            float pq = a0 * r[0] + a1 * r[1] + a2 * r[2] + a3 * r[3];
            for (int o = 16; o; o >>= 1) pq += __shfl_down_sync(0xffffffff, pq, o);
            if ((t & 31) == 0) hred[t >> 5][q] = pq;
        }
        __syncthreads();
        if (t < 32) {
            int nw = blockDim.x >> 5;
            const float *b1 = hp + H1 * 2 * H, *w2 = b1 + H1, *b2 = w2 + H2 * H1, *w3 = b2 + H2, *b3 = w3 + H2;
            float z1 = 0.0f;
            if (t < H1) {
                z1 = b1[t];
                for (int i = 0; i < nw; i++) z1 += hred[i][t];
                z1s[t] = z1;
            }
            __syncwarp();
            // Lane m: z2[m] and its L3 term.
            float z2 = b2[t];
#pragma unroll
            for (int k = 0; k < H1; k++) z2 += sq01(z1s[k]) * w2[t * H1 + k];
            float o3 = sq01(z2) * w3[t];
            for (int o = 16; o; o >>= 1) o3 += __shfl_down_sync(0xffffffff, o3, o);
            float g = 0.0f;
            if (t == 0) {
                float sum = 0;
                for (int i = 0; i < nw; i++) sum += red[i];
                float out = ob[s.bucket] + sum;
                out += b3[0] + o3;
                float pred = 1.0f / (1.0f + expf(-out));
                float err = pred - s.target;
                g = 2.0f * err * pred * (1.0f - pred);
                gout[sample] = g;
                loss[sample] = err * err;
            }
            g = __shfl_sync(0xffffffff, g, 0);
            // Head backward (as head_backward): dz2 = g W3 dsq(z2),
            // dz1 = dsq(z1) sum_m dz2[m] W2[m][k].
            float dz2 = g * w3[t] * dsq01(z2);
            dz2s[t] = dz2;
            __syncwarp();
            float *hsp = hs + (size_t)sample * HS;
            hsp[H1 + t] = z2;
            hsp[2 * H1 + H2 + t] = dz2;
            if (t < H1) {
                float sm = 0.0f;
                for (int m = 0; m < H2; m++) sm += dz2s[m] * w2[m * H1 + t];
                hsp[t] = z1;
                hsp[H1 + H2 + t] = sm * dsq01(z1);
            }
        }
        return;
    }
    __syncthreads();
    if (t == 0) {
        float sum = 0;
        for (int i = 0; i < (int)(blockDim.x >> 5); i++) sum += red[i];
        float out = ob[s.bucket] + sum;
        float pred = 1.0f / (1.0f + expf(-out));
        float err = pred - s.target;
        gout[sample] = 2.0f * err * pred * (1.0f - pred);
        loss[sample] = err * err;
    }
}

extern "C" __global__ void forward(const GSample *ss, const uint32_t *order, const float *wt, const float *rest,
                                   float *acc, float *gout, float *loss, int H, int buckets, float *hs, int hd) {
    forward_impl(ss, order, wt, rest, acc, gout, loss, H, buckets, hs, hd);
}

// `--half`: the same with the fp16 copy of the merged FT rows.
extern "C" __global__ void forward_half(const GSample *ss, const uint32_t *order, const __half *wt, const float *rest,
                                        float *acc, float *gout, float *loss, int H, int buckets, float *hs, int hd) {
    forward_impl(ss, order, wt, rest, acc, gout, loss, H, buckets, hs, hd);
}

// ---------------------------------------------------------------- FT backward

// Block = (32-column tile, work item = up to WI pairs of one king bucket),
// 1024 threads, lane = column; warp w owns the bucket's features w, w+32,
// ..., w+736 in registers. Per range of R pairs: stage dacc[R][32] in
// shared memory, set per-feature bitmasks of the pairs that contain it,
// then each warp adds its features' staged rows in pair order. Writes the
// item's gradient slice `part[item][768][H]` (zeros included).
#define R 128
#define FPW (INPUTS / 32)

// Per range of R pairs (ranges never cross a work item: items start at
// multiples of R within their bucket's run): the bitmask of the range's
// pairs that contain each of the bucket's features, `masks[range][768][R/32]`,
// built once per batch for every column tile.
extern "C" __global__ void plan_masks(const GSample *ss, const uint32_t *pairs, const int *range_start,
                                      const int *range_bucket, uint32_t *masks, int nranges) {
    __shared__ uint32_t mask[INPUTS][R / 32];
    int rg = blockIdx.x;
    for (int i = threadIdx.x; i < INPUTS * (R / 32); i += blockDim.x) (&mask[0][0])[i] = 0u;
    __syncthreads();
    int r0 = range_start[rg], nr = range_start[rg + 1] - r0, b = range_bucket[rg];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int r = warp; r < nr; r += nw) {
        uint32_t pr = pairs[r0 + r];
        const GSample &s = ss[pr >> 1];
        if (lane < s.n) atomicOr(&mask[s.f[pr & 1][lane] - b * INPUTS][r >> 5], 1u << (r & 31));
    }
    __syncthreads();
    uint32_t *out = masks + (size_t)rg * INPUTS * (R / 32);
    for (int i = threadIdx.x; i < INPUTS * (R / 32); i += blockDim.x) out[i] = (&mask[0][0])[i];
}

extern "C" __global__ void __launch_bounds__(1024) backward_ft(const GSample *ss, const uint32_t *pairs,
                                                              const Item *items, const uint32_t *masks,
                                                              const float *acc, const float *gout,
                                                              const float *rest, float *part, int H,
                                                              const float *hs, int hd, int buckets) {
    __shared__ float sd[2][R][32];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int col = blockIdx.x * 32 + lane;
    const float *ow = rest + H;
    Item it = items[blockIdx.y];
    float g[FPW];
#pragma unroll
    for (int i = 0; i < FPW; i++) g[i] = 0.0f;
    // Item.start/end are pair offsets; ranges of R pairs from it.start, the
    // global range index from it.range.
    int buf = 0;
    for (int r0 = it.start, rg = it.range; r0 < it.end; r0 += R, rg++, buf ^= 1) {
        int nr = min(R, it.end - r0);
        for (int r = warp; r < nr; r += 32) {
            uint32_t pr = pairs[r0 + r];
            int j = pr >> 1, side = pr & 1;
            const GSample &s = ss[j];
            float x = acc[((size_t)j * 2 + side) * H + col];
            float c = fminf(fmaxf(x, 0.f), 1.f);
            float w = ow[s.bucket * 2 * H + side * H + col];
            if (hd) {
                // As train_sample: e = sum_q dz1[q] W1[q][i], dacc = (g w + e) 2 a.
                const float *w1 = head_of(rest, H, buckets, s.bucket) + side * H + col;
                const float *dz = hs + (size_t)j * HS + H1 + H2;
                float e = 0.0f;
#pragma unroll
                for (int q = 0; q < H1; q++) e += dz[q] * w1[(size_t)q * 2 * H];
                sd[buf][r][lane] = (x > 0.f && x < 1.f) ? (gout[j] * w + e) * 2.0f * c : 0.0f;
            } else {
                sd[buf][r][lane] = (x > 0.f && x < 1.f) ? gout[j] * w * 2.0f * c : 0.0f;
            }
        }
        // Double-buffered staging: one barrier per range.
        __syncthreads();
        const uint32_t *mk = masks + (size_t)rg * INPUTS * (R / 32);
#pragma unroll
        for (int i = 0; i < FPW; i++) {
            const uint4 m4 = __ldg((const uint4 *)(mk + (warp + 32 * i) * (R / 32)));
            uint32_t ms[4] = {m4.x, m4.y, m4.z, m4.w};
#pragma unroll
            for (int q = 0; q < R / 32; q++) {
                uint32_t m = ms[q];
                while (m) {
                    int b = __ffs(m) - 1;
                    m &= m - 1;
                    g[i] += sd[buf][q * 32 + b][lane];
                }
            }
        }
    }
    float *out = part + (size_t)blockIdx.y * INPUTS * H;
#pragma unroll
    for (int i = 0; i < FPW; i++) out[(size_t)(warp + 32 * i) * H + col] = g[i];
}

// ---------------------------------------------------------------- rest backward

// Chunk of CH samples x 256 of the 2H (side, column) lanes: partial output-
// weight gradients per output bucket and partial FT-bias gradients, in
// sample order. pr layout per chunk: ow (B * 2H) | bias (2H) | ob (B).
#define CH 256
extern "C" __global__ void backward_rest(const GSample *ss, const float *acc, const float *gout, const float *rest,
                                         float *pr, int H, int buckets, int n, const float *hs, int hd) {
    int cc = blockIdx.x * blockDim.x + threadIdx.x;  // side * H + col
    int chunk = blockIdx.y;
    int j0 = chunk * CH, j1 = min(j0 + CH, n);
    size_t stride = (size_t)buckets * 2 * H + 2 * H + buckets;
    float *out = pr + chunk * stride;
    if (cc < 2 * H) {
        const float *ow = rest + H;
        float gw[8] = {0, 0, 0, 0, 0, 0, 0, 0}, gb = 0.0f;
        for (int j = j0; j < j1; j++) {
            int b = ss[j].bucket;
            float gg = gout[j];
            float x = acc[(size_t)j * 2 * H + cc];
            float c = fminf(fmaxf(x, 0.f), 1.f);
            float w = ow[b * 2 * H + cc];
#pragma unroll
            for (int k = 0; k < 8; k++)
                if (k == b) gw[k] += gg * c * c;
            if (hd) {
                const float *w1 = head_of(rest, H, buckets, b) + cc;
                const float *dz = hs + (size_t)j * HS + H1 + H2;
                float e = 0.0f;
#pragma unroll
                for (int q = 0; q < H1; q++) e += dz[q] * w1[(size_t)q * 2 * H];
                gb += (x > 0.f && x < 1.f) ? (gg * w + e) * 2.0f * c : 0.0f;
            } else {
                gb += (x > 0.f && x < 1.f) ? gg * w * 2.0f * c : 0.0f;
            }
        }
        for (int k = 0; k < buckets; k++) out[(size_t)k * 2 * H + cc] = gw[k];
        out[(size_t)buckets * 2 * H + cc] = gb;
    }
    if (blockIdx.x == 0 && threadIdx.x < buckets) {
        float s = 0.0f;
        for (int j = j0; j < j1; j++)
            if (ss[j].bucket == threadIdx.x) s += gout[j];
        out[(size_t)buckets * 2 * H + 2 * H + threadIdx.x] = s;
    }
}

// Sums the chunk partials in chunk order into the rest gradient.
extern "C" __global__ void reduce_rest(const float *pr, float *grest, int H, int buckets, int chunks) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)buckets * 2 * H + 2 * H + buckets;
    int nrest = H + buckets * 2 * H + buckets;
    if (i >= nrest) return;
    float s = 0.0f;
    if (i < H) {
        for (int k = 0; k < chunks; k++) {
            const float *b = pr + k * stride + (size_t)buckets * 2 * H;
            s += b[i];
            s += b[H + i];
        }
    } else if (i < H + buckets * 2 * H) {
        for (int k = 0; k < chunks; k++) s += pr[k * stride + (i - H)];
    } else {
        int b = i - H - buckets * 2 * H;
        for (int k = 0; k < chunks; k++) s += pr[k * stride + (size_t)buckets * 2 * H + 2 * H + b];
    }
    grest[i] = s;
}

// ---------------------------------------------------------------- head backward (--head)

// Partials of chunk `chunk` (CH samples) in the head's own layout:
// ph[chunk][bucket][head_len]. W1: thread = lane cc of 2H, grid.y = chunk *
// buckets + bucket; each thread walks the chunk's samples of its bucket in
// sample order (gW1[q][cc] += dz1[q] * a^2, as train_sample).
extern "C" __global__ void backward_head_w1(const GSample *ss, const float *acc, const float *hs, float *ph, int H,
                                            int buckets, int n) {
    int cc = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y % buckets, chunk = blockIdx.y / buckets;
    if (cc >= 2 * H) return;
    int j0 = chunk * CH, j1 = min(j0 + CH, n);
    float g[H1];
#pragma unroll
    for (int q = 0; q < H1; q++) g[q] = 0.0f;
    for (int j = j0; j < j1; j++) {
        if (ss[j].bucket != b) continue;
        float c = fminf(fmaxf(acc[(size_t)j * 2 * H + cc], 0.f), 1.f);
        float a2 = c * c;
        const float *dz = hs + (size_t)j * HS + H1 + H2;
#pragma unroll
        for (int q = 0; q < H1; q++) g[q] += dz[q] * a2;
    }
    float *out = ph + ((size_t)chunk * buckets + b) * head_len(H);
#pragma unroll
    for (int q = 0; q < H1; q++) out[(size_t)q * 2 * H + cc] = g[q];
}

// b1, W2, b2, W3, b3 (HSMALL params per bucket, thread = param), grid
// (buckets, chunks); as head_small_grads.
extern "C" __global__ void backward_head_small(const GSample *ss, const float *gout, const float *hs, float *ph, int H,
                                               int buckets, int n) {
    int b = blockIdx.x, chunk = blockIdx.y, t = threadIdx.x;
    if (t >= HSMALL) return;
    int j0 = chunk * CH, j1 = min(j0 + CH, n);
    float sm = 0.0f;
    for (int j = j0; j < j1; j++) {
        if (ss[j].bucket != b) continue;
        const float *z = hs + (size_t)j * HS;
        const float *z1 = z, *z2 = z + H1, *dz1 = z + H1 + H2, *dz2 = z + 2 * H1 + H2;
        float g = gout[j];
        if (t < H1) {
            sm += dz1[t];
        } else if (t < H1 + H2 * H1) {
            int m = (t - H1) / H1, k = (t - H1) % H1;
            sm += dz2[m] * sq01(z1[k]);
        } else if (t < H1 + H2 * H1 + H2) {
            sm += dz2[t - H1 - H2 * H1];
        } else if (t < H1 + H2 * H1 + 2 * H2) {
            sm += g * sq01(z2[t - H1 - H2 * H1 - H2]);
        } else {
            sm += g;
        }
    }
    ph[((size_t)chunk * buckets + b) * head_len(H) + H1 * 2 * H + t] = sm;
}

// Head gradient = the chunk partials summed in chunk order.
extern "C" __global__ void reduce_head(const float *ph, float *grest, int H, int buckets, int chunks) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t nh = (size_t)buckets * head_len(H);
    if (i >= nh) return;
    float sm = 0.0f;
    for (int k = 0; k < chunks; k++) sm += ph[(size_t)k * nh + i];
    grest[H + buckets * 2 * H + buckets + i] = sm;
}

// ---------------------------------------------------------------- AdamW

__device__ __forceinline__ void adam1(const AdamC &c, float decay, float &p, float &m, float &v, float g) {
    const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    g *= c.inv_batch;
    m = b1 * m + (1.0f - b1) * g;
    v = b2 * v + (1.0f - b2) * g * g;
    float upd = (m / c.c1) / (sqrtf(v / c.c2) + eps);
    p = fminf(fmaxf(p * decay - c.lr * upd, -CLIP), CLIP);
}

// Rest of the net (flat), gradient `g`.
extern "C" __global__ void adam_flat(AdamC c, float *p, float *m, float *v, const float *g, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float pp = p[i], mm = m[i], vv = v[i];
    adam1(c, c.decay, pp, mm, vv, g[i]);
    p[i] = pp;
    m[i] = mm;
    v[i] = vv;
}

// FT element (feature f, column col): bucket b's gradient = its work items'
// partials in item order (items ilo[b]..ilo[b+1]). With the factoriser
// (fac): F first from every bucket's gradient (bucket order), then each
// bucket's W, then the merged rows into wt. Without: each bucket's row.
// `active` (null = plain decay): rows without the batch's flag skip decay.
extern "C" __global__ void adam_ft(AdamC c, float *pw, float *mw, float *vw, float *pf, float *mf, float *vf,
                                   float *wt, __half *wth, const float *part, const int *ilo,
                                   const uint8_t *active, int H, int kb, int fac) {
    size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t nf = (size_t)INPUTS * H;
    if (e >= nf) return;
    int f = e / H;
    float gb[64];
    for (int b = 0; b < kb; b++) {
        float s = 0.0f;
        for (int k = ilo[b]; k < ilo[b + 1]; k++) s += part[(size_t)k * nf + e];
        gb[b] = s;
    }
    float fp = 0.0f;
    if (fac) {
        float gf = 0.0f;
        for (int b = 0; b < kb; b++) gf += gb[b];
        float pp = pf[e], mm = mf[e], vv = vf[e];
        adam1(c, c.decay, pp, mm, vv, gf);
        pf[e] = pp;
        mf[e] = mm;
        vf[e] = vv;
        fp = pp;
    }
    for (int b = 0; b < kb; b++) {
        size_t i = b * nf + e;
        float decay = (active && !active[b * INPUTS + f]) ? 1.0f : c.decay;
        float pp = pw[i], mm = mw[i], vv = vw[i];
        adam1(c, decay, pp, mm, vv, gb[b]);
        pw[i] = pp;
        mw[i] = mm;
        vw[i] = vv;
        float w = fac ? fminf(fmaxf(pp + fp, -CLIP), CLIP) : pp;
        if (wth)
            wth[i] = __float2half_rn(w);
        else
            wt[i] = w;
    }
}

// `--half`: fp16 copy of the merged FT rows (at import).
extern "C" __global__ void to_half(const float *src, __half *dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half_rn(src[i]);
}
