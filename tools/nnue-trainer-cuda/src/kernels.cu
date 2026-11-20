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
    // + 1/1024: exact ties of integer accumulators round up (nnue_trainer::PW_HALF).
    return floorf(__fmul_rn(p, PW_QX) + 0.5009765625f) / PW_QX;
}
__device__ __forceinline__ float sq01(float z) {
    float c = fminf(fmaxf(z, 0.f), 1.f);
    return c * c;
}
__device__ __forceinline__ float dsq01(float z) { return (z > 0.f && z < 1.f) ? 2.0f * z : 0.0f; }
// --qat-l2 (nnue_trainer::q_u1): the L1 activation as the engine's u8 / 127.
__device__ __forceinline__ float q_u1(float u) { return floorf(__fadd_rn(__fmul_rn(u, 127.0f), 0.5f)) * (1.0f / 127.0f); }

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
                                             float *acc, float *gout, float *loss, int H, int buckets, int hd,
                                             float *dacc) {
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
    const float wv[4] = {w[0], w[1], w[2], w[3]};
    float part = x0 * x0 * wv[0] + x1 * x1 * wv[1] + x2 * x2 * wv[2] + x3 * x3 * wv[3];
    for (int o = 16; o; o >>= 1) part += __shfl_down_sync(0xffffffff, part, o);
    __shared__ float red[32], gsh;
    if ((t & 31) == 0) red[t >> 5] = part;
    __syncthreads();
    if (t == 0) {
        float sum = 0;
        for (int i = 0; i < (int)(blockDim.x >> 5); i++) sum += red[i];
        if (hd) {
            // --head: the linear output's sum, for head_forward (which
            // overwrites gout with the gradient).
            gout[sample] = sum;
            return;
        }
        float out = ob[s.bucket] + sum;
        float pred = 1.0f / (1.0f + expf(-out));
        float err = pred - s.target;
        float g = 2.0f * err * pred * (1.0f - pred);
        gout[sample] = g;
        loss[sample] = err * err;
        gsh = g;
    }
    if (hd) return;
    // Plain: the FT backward's per-lane gradient dacc = g w 2 c for this
    // thread's 4 lanes, cached here (backward_ft's staging then reads one
    // value per pair instead of the sample, acc, w and g). Spelt out as
    // nvcc contracted it in backward_ft (2 g w as fma(g, w, g w), times c).
    __syncthreads();
    float g = gsh, xs[4] = {a.x, a.y, a.z, a.w}, d[4];
#pragma unroll
    for (int u = 0; u < 4; u++) {
        float x = xs[u], cl = fminf(fmaxf(x, 0.f), 1.f);
        float gw = __fmul_rn(g, wv[u]);
        d[u] = (x > 0.f && x < 1.f) ? __fmul_rn(cl, __fmaf_rn(g, wv[u], gw)) : 0.0f;
    }
    *(float4 *)(dacc + ((size_t)sample * 2 + side) * H + c) = make_float4(d[0], d[1], d[2], d[3]);
}

extern "C" __global__ void forward(const GSample *ss, const uint32_t *order, const float *wt, const float *rest,
                                   float *acc, float *gout, float *loss, int H, int buckets, int hd, float *dacc) {
    forward_impl(ss, order, wt, rest, acc, gout, loss, H, buckets, hd, dacc);
}

// `--half`: the same with the fp16 copy of the merged FT rows.
extern "C" __global__ void forward_half(const GSample *ss, const uint32_t *order, const __half *wt, const float *rest,
                                        float *acc, float *gout, float *loss, int H, int buckets, int hd, float *dacc) {
    forward_impl(ss, order, wt, rest, acc, gout, loss, H, buckets, hd, dacc);
}

// ---------------------------------------------------------------- head forward (--head)

// After `forward` (accumulators in acc, the linear output's sum in gout):
// one block of HT threads per sample. Same values and summation orders as
// the head code that used to run at the end of `forward` (and in
// compute_dacc), at a far higher occupancy than forward's H/2-thread
// blocks. The sample's accumulators are staged in shared memory, the head
// inputs (pw_in / sq01) formed once (also into `hin` for backward_head_w1),
// W1 read from quant_w1's aligned / lane-major copies. Then
// z1 (warp per row: lane partials over l = lane + 32 m, m ascending, then
// the shuffle tree), warp 0's L2 / L3 / loss / head backward, and dacc for
// every lane (as compute_dacc).
#define HT 256
extern "C" __global__ void __launch_bounds__(HT) head_forward(const GSample *ss, const float *acc, const float *rest,
                                                              const int8_t *qw, const float *wf, float *gout,
                                                              float *loss, float *hs,
                                                              float *hin, float *dacc, int H, int buckets, int hd,
                                                              float l1, float grp, int ql2) {
    int j = blockIdx.x, t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int nwarps = HT / 32;
    extern __shared__ float dsm[];
    float *as_s = dsm, *in_s = dsm + 2 * H;
    // b1 | W2 | b2 | W3 | b3 of the bucket; W2 also with rows padded to
    // H1 + 1 (no bank conflicts for the row reads of L2 or the column reads
    // of dz1).
    __shared__ float hsm[HSMALL], w2p[H2][H1 + 1], obs, tgs, lins, gs;
    __shared__ float z1s[H1], dz2s[H2], dz1s[H1];
    const GSample &s = ss[j];
    const int bk = s.bucket, nin = head_in(H, hd), half = H / 2;
    const float *ow = rest + H + (size_t)bk * 2 * H, *ob = rest + H + buckets * 2 * H;
    const float *w1 = head_of(rest, H, buckets, bk, hd);
    const int8_t *qb = qw + (size_t)bk * H1 * nin;
    {
        const float *src = w1 + H1 * nin;
        for (int k = t; k < HSMALL; k += HT) {
            float v = src[k];
            hsm[k] = v;
            int r = k - H1;
            // --qat-l2: L2 (forward and dz1) uses W2 as q_w1(W2).
            if (r >= 0 && r < H2 * H1) w2p[r / H1][r % H1] = ql2 ? q_w1(v) : v;
        }
        if (t == 0) {
            obs = ob[bk];
            tgs = s.target;
            lins = gout[j];
        }
        const float4 *ag = (const float4 *)(acc + (size_t)j * 2 * H);
        for (int k = t; k < H / 2; k += HT) ((float4 *)as_s)[k] = ag[k];
    }
    __syncthreads();
    {
        float *hrow = hin + (size_t)j * nin;
        for (int l = t; l < nin; l += HT) {
            float v;
            if (hd >= 2) {
                int sd = l >= half, i = l - sd * half;
                v = pw_in(as_s[sd * H + i], as_s[sd * H + i + half]);
            } else {
                v = sq01(as_s[l]);
            }
            in_s[l] = v;
            hrow[l] = v;
        }
    }
    __syncthreads();
    const float *b1 = hsm;
    for (int q = warp; q < H1; q += nwarps) {
        float z = 0.f;
        if (hd >= 2 && nin % 512 == 0) {
            // The lane's nin / 32 weights (lane-major copy) as 16-byte loads.
            const int K = nin / 32;
            const uint4 *r = (const uint4 *)(qw + (size_t)buckets * H1 * nin + (size_t)bk * H1 * nin + (size_t)q * nin +
                                             lane * K);
            for (int m0 = 0; m0 < K; m0 += 16) {
                uint4 u = r[m0 / 16];
                uint32_t wd[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
                for (int k = 0; k < 16; k++) {
                    int8_t kv = (int8_t)(wd[k / 4] >> (8 * (k % 4)));
                    z += in_s[lane + 32 * (m0 + k)] * ((float)kv * (1.0f / 64.0f));
                }
            }
        } else if (hd >= 2) {
            const int8_t *r = qb + (size_t)q * nin;
            for (int l = lane; l < nin; l += 32) z += in_s[l] * ((float)r[l] * (1.0f / 64.0f));
        } else if (nin % 128 == 0) {
            // Dense: the lane's nin / 32 weights (lane-major copy) as float4s.
            const int K = nin / 32;
            const float4 *r = (const float4 *)(wf + (size_t)buckets * H1 * nin + (size_t)bk * H1 * nin + (size_t)q * nin +
                                               lane * K);
            for (int m0 = 0; m0 < K; m0 += 4) {
                float4 u = r[m0 / 4];
                z += in_s[lane + 32 * m0] * u.x;
                z += in_s[lane + 32 * (m0 + 1)] * u.y;
                z += in_s[lane + 32 * (m0 + 2)] * u.z;
                z += in_s[lane + 32 * (m0 + 3)] * u.w;
            }
        } else {
            const float *r = w1 + (size_t)q * nin;
            for (int l = lane; l < nin; l += 32) z += in_s[l] * r[l];
        }
        for (int o = 16; o; o >>= 1) z += __shfl_down_sync(0xffffffff, z, o);
        if (lane == 0) z1s[q] = b1[q] + z;
    }
    __syncthreads();
    if (t < 32) {
        const float *w2 = b1 + H1, *b2 = w2 + H2 * H1, *w3 = b2 + H2, *b3 = w3 + H2;
        (void)w2;
        float z1 = (t < H1) ? z1s[t] : 0.f;
        // Lane m: z2[m] and its L3 term.
        float z2 = b2[t];
        const int ka = hd == 3 ? H1 - 1 : H1;
#pragma unroll
        if (ql2) {
            for (int k = 0; k < H1; k++)
                if (k < ka) z2 = __fadd_rn(z2, __fmul_rn(q_u1(sq01(z1s[k])), w2p[t][k]));
        } else {
            for (int k = 0; k < H1; k++)
                if (k < ka) z2 += sq01(z1s[k]) * w2p[t][k];
        }
        float o3 = sq01(z2) * w3[t];
        for (int o = 16; o; o >>= 1) o3 += __shfl_down_sync(0xffffffff, o3, o);
        float g = 0.0f;
        if (t == 0) {
            float out = obs + lins;
            out += b3[0] + o3;
            if (hd == 3) out += z1s[H1 - 1];
            float pred = 1.0f / (1.0f + expf(-out));
            float err = pred - tgs;
            g = 2.0f * err * pred * (1.0f - pred);
            gout[j] = g;
            loss[j] = err * err;
            gs = g;
        }
        g = __shfl_sync(0xffffffff, g, 0);
        // Head backward (as head_backward): dz2 = g W3 dsq(z2),
        // dz1 = dsq(z1) sum_m dz2[m] W2[m][k].
        float dz2 = g * w3[t] * dsq01(z2);
        dz2s[t] = dz2;
        __syncwarp();
        float *hsp = hs + (size_t)j * HS;
        hsp[H1 + t] = z2;
        hsp[2 * H1 + H2 + t] = dz2;
        if (t < H1) {
            float sm = 0.0f;
#pragma unroll
            for (int m = 0; m < H2; m++) sm += dz2s[m] * w2p[m][t];
            hsp[t] = z1;
            float d1 = (hd == 3 && t == H1 - 1) ? g : sm * dsq01(z1);
            hsp[H1 + H2 + t] = d1;
            dz1s[t] = d1;
        }
    }
    __syncthreads();
    // dacc for lanes cc .. cc + 3 (cc = side * H + col), as compute_dacc did.
    float gj = gs;
    if (hd >= 2) {
        // Pairwise: e depends only on the input pi = side * H/2 + col % (H/2),
        // shared by lane (side, col) and its partner col + H/2: sum it once
        // per 4 inputs, then form both lane groups.
        for (int g4 = t; g4 < H / 4; g4 += HT) {
            int pi = 4 * g4, side = pi >= half, c0 = pi - side * half;
            float e[4] = {0.f, 0.f, 0.f, 0.f};
            // --ft-l1-group (nnue_trainer::pw_group_dp): pi..pi+3 is one
            // aligned chunk; its extra dLoss/dp is grp p_u / ||p||.
            float gl[4] = {l1, l1, l1, l1};
            if (grp > 0.f) {
                float pr[4], n2 = 0.f;
#pragma unroll
                for (int u = 0; u < 4; u++) {
                    int a = side * H + c0 + u;
                    pr[u] = __fmul_rn(fminf(fmaxf(as_s[a], 0.f), 1.f), fminf(fmaxf(as_s[a + half], 0.f), 1.f));
                    n2 = __fadd_rn(n2, __fmul_rn(pr[u], pr[u]));
                }
                if (n2 > 0.f) {
                    float r = 1.0f / sqrtf(n2);
#pragma unroll
                    for (int u = 0; u < 4; u++) gl[u] = __fadd_rn(l1, __fmul_rn(grp, pr[u]) * r);
                }
            }
#pragma unroll
            for (int q = 0; q < H1; q++) {
                int k4 = *(const int *)(qb + (size_t)q * H + pi);
                float dq = dz1s[q];
#pragma unroll
                for (int u = 0; u < 4; u++) e[u] += dq * ((float)(int8_t)(k4 >> (8 * u)) * (1.0f / 64.0f));
            }
#pragma unroll
            for (int h2 = 0; h2 < 2; h2++) {
                int cc = side * H + c0 + h2 * half, pc = h2 ? cc - half : cc + half;
                float4 w4 = *(const float4 *)(ow + cc), a4 = *(const float4 *)(as_s + cc);
                float wo[4] = {w4.x, w4.y, w4.z, w4.w}, x[4] = {a4.x, a4.y, a4.z, a4.w}, d[4];
#pragma unroll
                for (int u = 0; u < 4; u++) {
                    float cl = fminf(fmaxf(x[u], 0.f), 1.f);
                    float cp = fminf(fmaxf(as_s[pc + u], 0.f), 1.f);
                    // The old compute_dacc's `g w 2 c + (e + l1) cp` exactly as nvcc
                    // contracted it there (2 g w as fma(g, w, g w)), spelt out so
                    // the different context cannot change the rounding.
                    float gw = __fmul_rn(wo[u], gj);
                    float m = __fmul_rn(cl, __fmaf_rn(wo[u], gj, gw));
                    d[u] = (x[u] > 0.f && x[u] < 1.f) ? __fmaf_rn(__fadd_rn(e[u], gl[u]), cp, m) : 0.0f;
                }
                *(float4 *)(dacc + (size_t)j * 2 * H + cc) = make_float4(d[0], d[1], d[2], d[3]);
            }
        }
        return;
    }
    for (int g4 = t; g4 < H / 2; g4 += HT) {
        int cc = 4 * g4;
        float4 w4 = *(const float4 *)(ow + cc), a4 = *(const float4 *)(as_s + cc);
        float wo[4] = {w4.x, w4.y, w4.z, w4.w}, x[4] = {a4.x, a4.y, a4.z, a4.w}, d[4];
        // The aligned copy of W1 (a bucket's head in `rest` is not 16-byte aligned).
        const float *w1c = wf + (size_t)bk * H1 * nin + cc;
        float e[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int q = 0; q < H1; q++) {
            float4 wq = *(const float4 *)(w1c + (size_t)q * 2 * H);
            float dq = dz1s[q];
            e[0] += dq * wq.x;
            e[1] += dq * wq.y;
            e[2] += dq * wq.z;
            e[3] += dq * wq.w;
        }
#pragma unroll
        for (int u = 0; u < 4; u++) {
            float cl = fminf(fmaxf(x[u], 0.f), 1.f);
            // `(g w + e) 2 c` as nvcc contracted it: fma(g, w, e), doubled, times c.
            float ge = __fmaf_rn(gj, wo[u], e[u]);
            d[u] = (x[u] > 0.f && x[u] < 1.f) ? __fmul_rn(cl, __fadd_rn(ge, ge)) : 0.0f;
        }
        *(float4 *)(dacc + (size_t)j * 2 * H + cc) = make_float4(d[0], d[1], d[2], d[3]);
    }
}

// ------------------------------------------------- W1 copies (--head)

// Once per batch before head_forward, two copies of every bucket's W1:
// [bucket][H1][nin] (16-byte aligned, unlike a bucket's head inside
// `rest`) for the dacc sums, and lane-major for the z1 sums (input
// l = lane + 32 m of row q at [bucket][q][lane][m], so a lane's weights
// are contiguous). Pairwise heads: int8 k = q_w1(w) * 64 in `qw` (|k| <=
// 127, so k / 64 is exactly q_w1(w)); dense head: f32 in `wf`.
extern "C" __global__ void quant_w1(const float *rest, int8_t *qw, float *wf, int H, int buckets, int hd) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int per = H1 * head_in(H, hd);
    if (i >= buckets * per) return;
    int b = i / per, k = i % per;
    float w = head_of(rest, H, buckets, b, hd)[k];
    int nin = head_in(H, hd), q = k / nin, l = k % nin, K = nin / 32;
    size_t lm = (size_t)buckets * per + (size_t)b * per + (size_t)q * nin + (l % 32) * K + l / 32;
    if (hd == 1) {
        wf[i] = w;
        wf[lm] = w;
        return;
    }
    int8_t v = (int8_t)fminf(fmaxf(roundf(w * 64.0f), -127.0f), 127.0f);
    qw[i] = v;
    qw[lm] = v;
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
            // Cached by forward / head_forward: same lane as acc[(j, side, col)].
            sd[r][lane] = dacc[((size_t)j * 2 + side) * H + col];
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
extern "C" __global__ void __launch_bounds__(256) backward_head_w1(const float *hin, const float *hs, float *ph,
                                                                    const uint32_t *bidx, const uint32_t *bcnt, int H,
                                                                    int buckets, int chunks, int hd) {
    int cc = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y % buckets, chunk = blockIdx.y / buckets;
    int nin = head_in(H, hd);
    int bi = b * chunks + chunk;
    const uint32_t *js = bidx + (size_t)bi * CH;
    int nj = bcnt[bi];
    // The list's dz1 rows and sample indices staged once for the block.
    __shared__ float dzs[CH][H1];
    __shared__ uint32_t jss[CH];
    for (int k = threadIdx.x; k < nj * H1; k += blockDim.x) {
        int z = k / H1, q = k % H1;
        dzs[z][q] = hs[(size_t)js[z] * HS + H1 + H2 + q];
    }
    for (int z = threadIdx.x; z < nj; z += blockDim.x) jss[z] = js[z];
    __syncthreads();
    if (cc >= nin) return;
    float g[H1];
#pragma unroll
    for (int q = 0; q < H1; q++) g[q] = 0.0f;
    for (int z = 0; z < nj; z++) {
        // The head input (pw_in / sq01) as the forward formed it.
        float a2 = hin[(size_t)jss[z] * nin + cc];
#pragma unroll
        for (int q = 0; q < H1; q++) g[q] += dzs[z][q] * a2;
    }
    float *out = ph + ((size_t)chunk * buckets + b) * head_len(H, hd);
#pragma unroll
    for (int q = 0; q < H1; q++) out[(size_t)q * nin + cc] = g[q];
}

// b1, W2, b2, W3, b3 (HSMALL params per bucket, thread = param), grid
// (buckets, chunks); the same per-bucket sample list as backward_head_w1.
#define HST 32
extern "C" __global__ void backward_head_small(const float *gout, const float *hs, float *ph, const uint32_t *bidx,
                                                const uint32_t *bcnt, int H, int buckets, int chunks, int hd, int ql2) {
    int b = blockIdx.x, chunk = blockIdx.y, t = threadIdx.x;
    int bi = b * chunks + chunk;
    const uint32_t *js = bidx + (size_t)bi * CH;
    int nj = bcnt[bi];
    // The list's hs rows and gout staged HST samples at a time (each thread
    // still sums its parameter's terms in list order).
    __shared__ float rows[HST][HS], gs[HST];
    float sm = 0.0f;
    for (int u0 = 0; u0 < nj; u0 += HST) {
        int nu = min(HST, nj - u0);
        __syncthreads();
        for (int k = t; k < nu * HS; k += blockDim.x) rows[k / HS][k % HS] = hs[(size_t)js[u0 + k / HS] * HS + k % HS];
        if (t < nu) gs[t] = gout[js[u0 + t]];
        __syncthreads();
        if (t >= HSMALL) continue;
        for (int u = 0; u < nu; u++) {
            const float *z = rows[u];
            const float *z1 = z, *z2 = z + H1, *dz1 = z + H1 + H2, *dz2 = z + 2 * H1 + H2;
            float g = gs[u];
            if (t < H1) {
                sm += dz1[t];
            } else if (t < H1 + H2 * H1) {
                int m = (t - H1) / H1, k = (t - H1) % H1;
                // Pairwise-skip: the skip neuron does not feed L2.
                if (!(hd == 3 && k == H1 - 1)) sm += dz2[m] * (ql2 ? q_u1(sq01(z1[k])) : sq01(z1[k]));
            } else if (t < H1 + H2 * H1 + H2) {
                sm += dz2[t - H1 - H2 * H1];
            } else if (t < H1 + H2 * H1 + 2 * H2) {
                sm += g * sq01(z2[t - H1 - H2 * H1 - H2]);
            } else {
                sm += g;
            }
        }
    }
    if (t < HSMALL) ph[((size_t)chunk * buckets + b) * head_len(H, hd) + H1 * head_in(H, hd) + t] = sm;
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
