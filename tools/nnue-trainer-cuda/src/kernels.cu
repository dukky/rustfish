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
                                             float *acc, float *gout, float *loss, int H, int buckets) {
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
                                   float *acc, float *gout, float *loss, int H, int buckets) {
    forward_impl(ss, order, wt, rest, acc, gout, loss, H, buckets);
}

// `--half`: the same with the fp16 copy of the merged FT rows.
extern "C" __global__ void forward_half(const GSample *ss, const uint32_t *order, const __half *wt, const float *rest,
                                        float *acc, float *gout, float *loss, int H, int buckets) {
    forward_impl(ss, order, wt, rest, acc, gout, loss, H, buckets);
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
                                                              const float *rest, float *part, int H) {
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
            sd[buf][r][lane] = (x > 0.f && x < 1.f) ? gout[j] * w * 2.0f * c : 0.0f;
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
                                         float *pr, int H, int buckets, int n) {
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
            gb += (x > 0.f && x < 1.f) ? gg * w * 2.0f * c : 0.0f;
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
