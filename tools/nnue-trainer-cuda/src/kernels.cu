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
// hd: 1 = dense head (2H SCReLU inputs), 2 = pairwise (H inputs, see
// nnue_trainer::pw_in; linear SCReLU output kept), 3 = pairwise-skip (no
// linear output; L1 neuron 15 is added to the output, 0..15 feed L2).
// Pairwise heads run the fake-quantised forward: W1 as q_w1, inputs as
// pw_in (the CPU trainer's functions).
#define PW_QX (65025.0f / 512.0f)
__device__ __forceinline__ int head_in(int H, int hd) { return hd >= 2 ? H : 2 * H; }
__device__ __forceinline__ int head_len(int H, int hd) { return H1 * head_in(H, hd) + HSMALL; }
__device__ __forceinline__ const float *head_of(const float *rest, int H, int buckets, int b, int hd) {
    return rest + H + buckets * 2 * H + buckets + (size_t)b * head_len(H, hd);
}
__device__ __forceinline__ float q_w1(float w) { return fminf(fmaxf(roundf(w * 64.0f), -127.0f), 127.0f) * (1.0f / 64.0f); }
__device__ __forceinline__ float pw_in(float a, float b) {
    float p = fminf(fmaxf(a, 0.f), 1.f) * fminf(fmaxf(b, 0.f), 1.f);
    // No fma contraction: round the product first, as the CPU trainer.
    return floorf(__fmul_rn(p, PW_QX) + 0.5f) / PW_QX;
}
// Pairwise input i (0..H) of one sample's accumulators `as` (2H).
__device__ __forceinline__ float pw_at(const float *as, int H, int i) {
    int half = H / 2, s = i / half, j = i % half;
    return pw_in(as[s * H + j], as[s * H + j + half]);
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
        __syncthreads();
        __shared__ float z1s[H1], dz2s[H2];
        int lane = t & 31, warp = t >> 5, nwarps = blockDim.x >> 5;
        const float *hp = head_of(rest, H, buckets, s.bucket, hd);
        const int nin = head_in(H, hd);
        const float *w1 = hp;
        const float *b1 = hp + H1 * nin;
        const float *as = acc + (size_t)sample * 2 * H;
        for (int q = warp; q < H1; q += nwarps) {
            float z = 0.f;
            const float *r = w1 + (size_t)q * nin;
            if (hd >= 2) {
                for (int l = lane; l < nin; l += 32) z += pw_at(as, H, l) * q_w1(r[l]);
            } else {
                for (int l = lane; l < nin; l += 32) z += sq01(as[l]) * r[l];
            }
            for (int o = 16; o; o >>= 1) z += __shfl_down_sync(0xffffffff, z, o);
            if (lane == 0) z1s[q] = b1[q] + z;
        }
        __syncthreads();
        if (t < 32) {
            int nw = nwarps;
            const float *w2 = b1 + H1, *b2 = w2 + H2 * H1, *w3 = b2 + H2, *b3 = w3 + H2;
            float z1 = (t < H1) ? z1s[t] : 0.f;
            __syncwarp();
            // Lane m: z2[m] and its L3 term.
            float z2 = b2[t];
            const int ka = hd == 3 ? H1 - 1 : H1;
            for (int k = 0; k < ka; k++) z2 += sq01(z1s[k]) * w2[t * H1 + k];
            float o3 = sq01(z2) * w3[t];
            for (int o = 16; o; o >>= 1) o3 += __shfl_down_sync(0xffffffff, o3, o);
            float g = 0.0f;
            if (t == 0) {
                float sum = 0;
                for (int i = 0; i < nw; i++) sum += red[i];
                float out = ob[s.bucket] + sum;
                out += b3[0] + o3;
                if (hd == 3) out += z1s[H1 - 1];
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
                hsp[H1 + H2 + t] = (hd == 3 && t == H1 - 1) ? g : sm * dsq01(z1);
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

// --------------------------------------------------------- cached dacc (--head)

// Both backward branches need the same dacc[(j, cc)] for every sample j and
// flattened lane cc = side * H + col: cache it here instead of redoing the
// dense 16-term e sum twice. Same expression and operation order as the old
// head branches; sample-major/lane-minor order, no atomics.
extern "C" __global__ void __launch_bounds__(256) compute_dacc(const GSample *ss, const float *acc,
                                                               const float *gout, const float *rest,
                                                               const float *hs, float *dacc, int H,
                                                               int buckets, int n, int hd, float l1) {
    int cc = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y;
    if (cc >= 2 * H || j >= n) return;
    size_t at = (size_t)j * 2 * H + cc;
    int b = ss[j].bucket;
    float x = acc[at];
    float c = fminf(fmaxf(x, 0.f), 1.f);
    const float *ow = rest + H;
    float w = ow[b * 2 * H + cc];
    const float *dz = hs + (size_t)j * HS + H1 + H2;
    if (hd >= 2) {
        // Pairwise: lane (side, col) feeds input side * H/2 + col % (H/2)
        // with partner col +- H/2 (nnue_trainer::pw_lane / pw_dacc).
        int half = H / 2, side = cc / H, col = cc % H;
        int pi = side * half + col % half;
        int partner = col < half ? cc + half : cc - half;
        const float *w1 = head_of(rest, H, buckets, b, hd) + pi;
        float e = 0.0f;
#pragma unroll
        for (int q = 0; q < H1; q++) e += dz[q] * q_w1(w1[(size_t)q * H]);
        float cp = fminf(fmaxf(acc[(size_t)j * 2 * H + partner], 0.f), 1.f);
        dacc[at] = (x > 0.f && x < 1.f) ? gout[j] * w * 2.0f * c + (e + l1) * cp : 0.0f;
        return;
    }
    const float *w1 = head_of(rest, H, buckets, b, hd) + cc;
    float e = 0.0f;
#pragma unroll
    for (int q = 0; q < H1; q++) e += dz[q] * w1[(size_t)q * 2 * H];
    dacc[at] = (x > 0.f && x < 1.f) ? (gout[j] * w + e) * 2.0f * c : 0.0f;
}

// ---------------------------------------------------------------- FT backward

// Ranges of R pairs (they never cross a work item: items start at
// multiples of R within their bucket's run). `plan_csr`, once per batch for
// every column tile: per range, its incidences (pair, feature) sorted by
// feature, pairs ascending within a feature: `csr[range]` = per feature a
// packed (start | end << 16) into the range's list, then the list of pair
// offsets in the range (bytes, < R).
#define R 256
#define RW (R / 32)
#define LIST_WORDS (R * 32 / 4)
#define CSRW (INPUTS + LIST_WORDS)

extern "C" __global__ void __launch_bounds__(256) plan_csr(const GSample *ss, const uint32_t *pairs, const int *range_start,
                                                           const int *range_bucket, uint32_t *csr) {
    __shared__ uint32_t mask[INPUTS][RW];
    __shared__ int wsum[8];
    int rg = blockIdx.x, t = threadIdx.x;
    for (int i = t; i < INPUTS * RW; i += 256) (&mask[0][0])[i] = 0u;
    __syncthreads();
    int r0 = range_start[rg], nr = range_start[rg + 1] - r0, b = range_bucket[rg];
    int lane = t & 31, warp = t >> 5;
    for (int r = warp; r < nr; r += 8) {
        uint32_t pr = pairs[r0 + r];
        const GSample &s = ss[pr >> 1];
        if (lane < s.n) atomicOr(&mask[s.f[pr & 1][lane] - b * INPUTS][r >> 5], 1u << (r & 31));
    }
    __syncthreads();
    // Thread t: features 3t .. 3t + 2; exclusive scan of their counts.
    int c[3], tot = 0;
#pragma unroll
    for (int j = 0; j < 3; j++) {
        int n = 0;
#pragma unroll
        for (int q = 0; q < RW; q++) n += __popc(mask[3 * t + j][q]);
        c[j] = n;
        tot += n;
    }
    int x = tot;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        int y = __shfl_up_sync(0xffffffff, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) wsum[warp] = x;
    __syncthreads();
    int base = x - tot;
    for (int w = 0; w < warp; w++) base += wsum[w];
    uint32_t *out = csr + (size_t)rg * CSRW;
    uint8_t *lst = (uint8_t *)(out + INPUTS);
#pragma unroll
    for (int j = 0; j < 3; j++) {
        int f = 3 * t + j, k = base;
        out[f] = (uint32_t)base | (uint32_t)(base + c[j]) << 16;
#pragma unroll
        for (int q = 0; q < RW; q++) {
            uint32_t m = mask[f][q];
            while (m) {
                int bit = __ffs(m) - 1;
                m &= m - 1;
                lst[k++] = (uint8_t)(q * 32 + bit);
            }
        }
        base += c[j];
    }
}

// Block = (32-column tile, work item = up to WI pairs of one king bucket),
// 1024 threads, lane = column; warp w owns the bucket's features w, w+32,
// ..., w+736 in registers. Per range of R pairs: stage dacc[R][32] and the
// range's CSR in shared memory, then each warp adds its features' staged
// rows in pair order (the same order as a scan of per-feature pair
// bitmasks). Writes the item's gradient slice `part[item][768][H]` (zeros
// included).
#define FPW (INPUTS / 32)
extern "C" __global__ void __launch_bounds__(1024) backward_ft(const GSample *ss, const uint32_t *pairs,
                                                               const Item *items, const uint32_t *csr,
                                                               const float *acc, const float *gout,
                                                               const float *rest, float *part, int H,
                                                               const float *hs, const float *dacc, int hd, int buckets) {
    __shared__ float sd[R][32];
    __shared__ uint32_t so[INPUTS];
    __shared__ uint32_t sl[LIST_WORDS];
    int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    int col = blockIdx.x * 32 + lane;
    const float *ow = rest + H;
    Item it = items[blockIdx.y];
    float g[FPW];
#pragma unroll
    for (int i = 0; i < FPW; i++) g[i] = 0.0f;
    for (int r0 = it.start, rg = it.range; r0 < it.end; r0 += R, rg++) {
        int nr = min(R, it.end - r0);
        // Single-buffered (R = 256 fills the 48 KB): the previous range's
        // sums are done before its staging is overwritten.
        __syncthreads();
        for (int r = warp; r < nr; r += 32) {
            uint32_t pr = pairs[r0 + r];
            int j = pr >> 1, side = pr & 1;
            const GSample &s = ss[j];
            float x = acc[((size_t)j * 2 + side) * H + col];
            float c = fminf(fmaxf(x, 0.f), 1.f);
            float w = ow[s.bucket * 2 * H + side * H + col];
            if (hd) {
                // Cached by compute_dacc: same lane as acc[(j, side, col)].
                sd[r][lane] = dacc[((size_t)j * 2 + side) * H + col];
            } else {
                sd[r][lane] = (x > 0.f && x < 1.f) ? gout[j] * w * 2.0f * c : 0.0f;
            }
        }
        const uint32_t *cs = csr + (size_t)rg * CSRW;
        if (t < INPUTS) so[t] = cs[t];
        int nw = ((cs[INPUTS - 1] >> 16) + 3) >> 2;
        for (int w = t; w < nw; w += 1024) sl[w] = cs[INPUTS + w];
        __syncthreads();
        const uint8_t *lb = (const uint8_t *)sl;
#pragma unroll
        for (int i = 0; i < FPW; i++) {
            uint32_t oe = so[warp + 32 * i];
            for (int k = oe & 0xffff, e = oe >> 16; k < e; k++) g[i] += sd[lb[k]][lane];
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
                                         float *pr, int H, int buckets, int n, const float *hs, const float *dacc,
                                         int hd) {
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
                gb += dacc[(size_t)j * 2 * H + cc];
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
extern "C" __global__ void reduce_rest(const float *pr, float *grest, int H, int buckets, int chunks, int hd) {
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
        // Pairwise-skip has no linear output: its out weights stay 0.
        if (hd != 3)
            for (int k = 0; k < chunks; k++) s += pr[k * stride + (i - H)];
    } else {
        int b = i - H - buckets * 2 * H;
        for (int k = 0; k < chunks; k++) s += pr[k * stride + (size_t)buckets * 2 * H + 2 * H + b];
    }
    grest[i] = s;
}

// ---------------------------------------------------------------- head backward (--head)

// Partials of chunk `chunk` (up to CH samples) in the head's own layout:
// ph[chunk][bucket][head_len]. W1: thread = lane cc of 2H, grid.y = chunk *
// buckets + bucket; bidx[bucket, chunk] lists that chunk's samples of the
// bucket in sample order, so the sum matches the old bucket-filtered scan.
extern "C" __global__ void backward_head_w1(const float *acc, const float *hs, float *ph, const uint32_t *bidx,
                                             const uint32_t *bcnt, int H, int buckets, int chunks, int hd) {
    int cc = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y % buckets, chunk = blockIdx.y / buckets;
    int nin = head_in(H, hd);
    if (cc >= nin) return;
    int bi = b * chunks + chunk;
    const uint32_t *js = bidx + (size_t)bi * CH;
    int nj = bcnt[bi];
    float g[H1];
#pragma unroll
    for (int q = 0; q < H1; q++) g[q] = 0.0f;
    for (int z = 0; z < nj; z++) {
        int j = js[z];
        float a2;
        if (hd >= 2) {
            a2 = pw_at(acc + (size_t)j * 2 * H, H, cc);
        } else {
            float c = fminf(fmaxf(acc[(size_t)j * 2 * H + cc], 0.f), 1.f);
            a2 = c * c;
        }
        const float *dz = hs + (size_t)j * HS + H1 + H2;
#pragma unroll
        for (int q = 0; q < H1; q++) g[q] += dz[q] * a2;
    }
    float *out = ph + ((size_t)chunk * buckets + b) * head_len(H, hd);
#pragma unroll
    for (int q = 0; q < H1; q++) out[(size_t)q * nin + cc] = g[q];
}

// b1, W2, b2, W3, b3 (HSMALL params per bucket, thread = param), grid
// (buckets, chunks); the same per-bucket sample list as backward_head_w1.
extern "C" __global__ void backward_head_small(const float *gout, const float *hs, float *ph, const uint32_t *bidx,
                                                const uint32_t *bcnt, int H, int buckets, int chunks, int hd) {
    int b = blockIdx.x, chunk = blockIdx.y, t = threadIdx.x;
    if (t >= HSMALL) return;
    int bi = b * chunks + chunk;
    const uint32_t *js = bidx + (size_t)bi * CH;
    int nj = bcnt[bi];
    float sm = 0.0f;
    for (int u = 0; u < nj; u++) {
        int j = js[u];
        const float *z = hs + (size_t)j * HS;
        const float *z1 = z, *z2 = z + H1, *dz1 = z + H1 + H2, *dz2 = z + 2 * H1 + H2;
        float g = gout[j];
        if (t < H1) {
            sm += dz1[t];
        } else if (t < H1 + H2 * H1) {
            int m = (t - H1) / H1, k = (t - H1) % H1;
            // Pairwise-skip: the skip neuron does not feed L2.
            if (!(hd == 3 && k == H1 - 1)) sm += dz2[m] * sq01(z1[k]);
        } else if (t < H1 + H2 * H1 + H2) {
            sm += dz2[t - H1 - H2 * H1];
        } else if (t < H1 + H2 * H1 + 2 * H2) {
            sm += g * sq01(z2[t - H1 - H2 * H1 - H2]);
        } else {
            sm += g;
        }
    }
    ph[((size_t)chunk * buckets + b) * head_len(H, hd) + H1 * head_in(H, hd) + t] = sm;
}

// Head gradient = the chunk partials summed in chunk order.
extern "C" __global__ void reduce_head(const float *ph, float *grest, int H, int buckets, int chunks, int hd) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t nh = (size_t)buckets * head_len(H, hd);
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
