// Quick CUDA throughput bench for the (KB*768 -> H) x 2 -> B SCReLU nets of
// tools/nnue-trainer: forward, backward (float atomics: fast, but not
// deterministic, unlike the CPU trainer), AdamW (+ factoriser merge).
// Same decode, init, loss and AdamW maths as the CPU trainer.
//
//   nvcc -O3 -arch=sm_61 -o bench bench.cu
//   ./bench <records.bin> [--hidden 512] [--buckets 8] [--kb kb8m|none]
//           [--batches 200] [--batch 16384]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

static const int INPUTS = 768;
static const float SCALE = 400.0f, CLIP = 1.98f;

// ---------------------------------------------------------------- host decode (mirrors main.rs)

static const uint8_t KB8M_HALF[32] = {0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6,
                                      7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7};

struct Layout {
    uint8_t b[64];
    bool mirror;
    int kb;
};

static Layout make_layout(const std::string &name) {
    Layout l{};
    if (name == "none") {
        l.kb = 1;
    } else if (name == "kb8m") {
        for (int i = 0; i < 32; i++) {
            int r = i / 4, f = i % 4;
            l.b[r * 8 + f] = l.b[r * 8 + 7 - f] = KB8M_HALF[i];
        }
        l.mirror = true;
        l.kb = 8;
    } else {
        fprintf(stderr, "layout %s not in the bench\n", name.c_str());
        exit(1);
    }
    return l;
}

struct Sample {
    uint16_t f[2][32];
    uint8_t n, bucket;
    float target;
};

static float sigm(float x) { return 1.0f / (1.0f + expf(-x)); }

static Sample decode(const uint8_t *r, float lambda, int buckets, bool mirror, const Layout &kl) {
    uint64_t occ0;
    memcpy(&occ0, r, 8);
    int16_t score;
    memcpy(&score, r + 24, 2);
    bool stm_black = r[27] == 1;
    Sample s{};
    uint16_t flip = mirror ? 7 : 0;
    uint16_t ctx_off[2] = {0, 0}, ctx_x[2] = {0, 0};
    if (kl.kb > 1 || kl.mirror) {
        uint64_t occ = occ0;
        int n = 0;
        uint16_t ksq[2] = {0, 0};
        while (occ) {
            uint16_t sq = (uint16_t)__builtin_ctzll(occ) ^ flip;
            occ &= occ - 1;
            uint8_t nib = r[8 + n / 2] >> (4 * (n & 1)) & 0xF;
            if ((nib & 7) == 5) ksq[nib >> 3] = sq;
            n++;
        }
        bool persp[2] = {stm_black, !stm_black};
        for (int i = 0; i < 2; i++) {
            uint16_t k = ksq[persp[i]];
            uint16_t rel = persp[i] ? k ^ 56 : k;
            ctx_x[i] = (kl.mirror && (rel & 7) >= 4) ? 7 : 0;
            ctx_off[i] = kl.b[rel] * INPUTS;
        }
    }
    uint64_t occ = occ0;
    while (occ) {
        uint16_t sq = (uint16_t)__builtin_ctzll(occ) ^ flip;
        occ &= occ - 1;
        uint8_t nib = r[8 + s.n / 2] >> (4 * (s.n & 1)) & 0xF;
        bool black = nib >> 3 == 1;
        uint16_t kind = nib & 7;
        bool persp[2] = {stm_black, !stm_black};
        for (int i = 0; i < 2; i++) {
            uint16_t side = black == persp[i] ? 0 : 384;
            uint16_t rel = persp[i] ? sq ^ 56 : sq;
            s.f[i][s.n] = ctx_off[i] + side + kind * 64 + (rel ^ ctx_x[i]);
        }
        s.n++;
    }
    s.bucket = buckets == 1 ? 0 : std::min((std::max(s.n, (uint8_t)2) - 2) * buckets / 32, buckets - 1);
    float wdl = r[26] / 2.0f, sc = score;
    if (stm_black) {
        wdl = 1.0f - wdl;
        sc = -sc;
    }
    s.target = lambda * wdl + (1.0f - lambda) * sigm(sc / SCALE);
    return s;
}

struct Rng {
    uint64_t s;
    uint64_t next() {
        s += 0x9E3779B97F4A7C15ull;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    float uniform(float a) { return ((float)(next() >> 40) / (float)(1u << 24) * 2.0f - 1.0f) * a; }
};

// ---------------------------------------------------------------- kernels

// One block per sample, H/2 threads: thread t owns side t / (H/4) and four
// columns. Stores the accumulators, the output gradient and the loss.
__global__ void forward(const Sample *ss, const uint32_t *order, const float *wt, const float *p_ftb, const float *p_ow,
                        const float *p_ob, float *acc, float *gout, float *loss, int H) {
    int sample = order ? order[blockIdx.x] : blockIdx.x;
    const Sample &s = ss[sample];
    int q = H / 4, t = threadIdx.x, side = t / q, c = (t % q) * 4;
    float4 a = *(const float4 *)(p_ftb + c);
    int n = s.n;
    for (int i = 0; i < n; i++) {
        float4 w = __ldg((const float4 *)(wt + (size_t)s.f[side][i] * H + c));
        a.x += w.x, a.y += w.y, a.z += w.z, a.w += w.w;
    }
    *(float4 *)(acc + ((size_t)sample * 2 + side) * H + c) = a;
    const float *ow = p_ow + s.bucket * 2 * H + side * H + c;
    float x0 = fminf(fmaxf(a.x, 0.f), 1.f), x1 = fminf(fmaxf(a.y, 0.f), 1.f), x2 = fminf(fmaxf(a.z, 0.f), 1.f),
          x3 = fminf(fmaxf(a.w, 0.f), 1.f);
    float part = x0 * x0 * ow[0] + x1 * x1 * ow[1] + x2 * x2 * ow[2] + x3 * x3 * ow[3];
    for (int o = 16; o; o >>= 1) part += __shfl_down_sync(0xffffffff, part, o);
    __shared__ float red[32];
    if ((t & 31) == 0) red[t >> 5] = part;
    __syncthreads();
    if (t == 0) {
        float sum = 0;
        for (int i = 0; i < (int)(blockDim.x >> 5); i++) sum += red[i];
        float out = p_ob[s.bucket] + sum;
        float pred = 1.0f / (1.0f + expf(-out));
        float err = pred - s.target;
        gout[sample] = 2.0f * err * pred * (1.0f - pred);
        loss[sample] = err * err;
    }
}

// SPB samples per block; FT row gradients by float atomics, FT bias and
// output-layer gradients summed per block first.
template <int SPB>
__global__ void backward(const Sample *ss, const float *acc, const float *gout, const float *p_ow, float *g_ft,
                         float *g_ftb, float *g_ow, float *g_ob, int H, int n_samples, int buckets) {
    int q = H / 4, t = threadIdx.x, side = t / q, c = (t % q) * 4;
    float4 gb = {0, 0, 0, 0};
    float4 gw[8];
    for (int b = 0; b < 8; b++) gw[b] = {0, 0, 0, 0};
    float gob[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int s0 = blockIdx.x * SPB;
    for (int j = s0; j < min(s0 + SPB, n_samples); j++) {
        const Sample &s = ss[j];
        float g = gout[j];
        float4 x = *(const float4 *)(acc + ((size_t)j * 2 + side) * H + c);
        const float *ow = p_ow + s.bucket * 2 * H + side * H + c;
        float xs[4] = {x.x, x.y, x.z, x.w}, d[4];
        float cw[4];
        for (int k = 0; k < 4; k++) {
            float cl = fminf(fmaxf(xs[k], 0.f), 1.f);
            cw[k] = g * cl * cl;
            d[k] = (xs[k] > 0.f && xs[k] < 1.f) ? g * ow[k] * 2.0f * cl : 0.0f;
        }
        int b = s.bucket;
        // (bucket is uniform per sample: no divergence)
#pragma unroll
        for (int bb = 0; bb < 8; bb++)
            if (bb == b) gw[bb].x += cw[0], gw[bb].y += cw[1], gw[bb].z += cw[2], gw[bb].w += cw[3];
        gb.x += d[0], gb.y += d[1], gb.z += d[2], gb.w += d[3];
        if (t == 0) gob[b] += g;
        for (int i = 0; i < s.n; i++) {
            float *row = g_ft + (size_t)s.f[side][i] * H + c;
            atomicAdd(row + 0, d[0]);
            atomicAdd(row + 1, d[1]);
            atomicAdd(row + 2, d[2]);
            atomicAdd(row + 3, d[3]);
        }
    }
    atomicAdd(g_ftb + c + 0, gb.x);
    atomicAdd(g_ftb + c + 1, gb.y);
    atomicAdd(g_ftb + c + 2, gb.z);
    atomicAdd(g_ftb + c + 3, gb.w);
    for (int b = 0; b < buckets; b++) {
        float *w = g_ow + b * 2 * H + side * H + c;
        atomicAdd(w + 0, gw[b].x);
        atomicAdd(w + 1, gw[b].y);
        atomicAdd(w + 2, gw[b].z);
        atomicAdd(w + 3, gw[b].w);
    }
    if (t == 0)
        for (int b = 0; b < buckets; b++) atomicAdd(g_ob + b, gob[b]);
}


// Tiled backward (the Metal trainer's grad_fused): block = (32-column
// tile, work item = up to WI pairs of one king bucket), 1024 threads, lane =
// column, warp w owns that bucket's features w, w+32, ..., w+736 in
// registers. Per range of R pairs: stage dacc[R][32] in shared memory
// (collecting the FT-bias and output-weight gradients from the same
// reads), set a per-feature bitmask of the pairs that contain it, then each
// warp adds the staged rows of its features in pair order. The bucket's
// gradient slice is written once per block (atomics across work items).
#define R 128
#define FPW (INPUTS / 32)
struct Item {
    int bucket, start, end;
};

__global__ void __launch_bounds__(1024) backward_tiled(const Sample *ss, const uint32_t *pairs, const Item *items,
                                                      const float *acc, const float *gout, const float *p_ow,
                                                      float *g_ft, float *g_ftb, float *g_ow, float *g_ob, int H,
                                                      int buckets) {
    __shared__ float sd[R][32];
    __shared__ uint32_t mask[INPUTS][R / 32];
    __shared__ float sgow[16][32];
    __shared__ float sgb[32];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int col = blockIdx.x * 32 + lane;
    Item it = items[blockIdx.y];
    float g[FPW];
#pragma unroll
    for (int i = 0; i < FPW; i++) g[i] = 0.0f;
    for (int i = threadIdx.x; i < 16 * 32; i += 1024) (&sgow[0][0])[i] = 0.0f;
    if (threadIdx.x < 32) sgb[threadIdx.x] = 0.0f;
    float my_gb = 0.0f;
    for (int r0 = it.start; r0 < it.end; r0 += R) {
        int nr = min(R, it.end - r0);
        for (int i = threadIdx.x; i < INPUTS * (R / 32); i += 1024) (&mask[0][0])[i] = 0u;
        __syncthreads();
        // Warp w stages pairs w, w+32, ...: dacc and its feature bits.
        for (int r = warp; r < nr; r += 32) {
            uint32_t pr = pairs[r0 + r];
            int j = pr >> 1, side = pr & 1;
            const Sample &s = ss[j];
            float x = acc[((size_t)j * 2 + side) * H + col];
            float gg = gout[j];
            float c = fminf(fmaxf(x, 0.f), 1.f);
            float w = p_ow[s.bucket * 2 * H + side * H + col];
            float d = (x > 0.f && x < 1.f) ? gg * w * 2.0f * c : 0.0f;
            sd[r][lane] = d;
            my_gb += d;
            atomicAdd(&sgow[s.bucket * 2 + side][lane], gg * c * c);
            if (blockIdx.x == 0 && lane == 0 && side == 0) atomicAdd(&g_ob[s.bucket], gg);
            if (lane < s.n) {
                int f = s.f[side][lane] - it.bucket * INPUTS;
                atomicOr(&mask[f][r >> 5], 1u << (r & 31));
            }
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < FPW; i++) {
            int f = warp + 32 * i;
#pragma unroll
            for (int q = 0; q < R / 32; q++) {
                uint32_t m = mask[f][q];
                while (m) {
                    int b = __ffs(m) - 1;
                    m &= m - 1;
                    g[i] += sd[q * 32 + b][lane];
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < FPW; i++) {
        if (g[i] != 0.0f) atomicAdd(&g_ft[((size_t)it.bucket * INPUTS + warp + 32 * i) * H + col], g[i]);
    }
    atomicAdd(&sgb[lane], my_gb);
    __syncthreads();
    for (int i = threadIdx.x; i < buckets * 2 * 32; i += 1024) {
        int bs = i / 32, l = i % 32;
        atomicAdd(&g_ow[(bs >> 1) * 2 * H + (bs & 1) * H + blockIdx.x * 32 + l], sgow[bs][l]);
    }
    if (threadIdx.x < 32) atomicAdd(&g_ftb[blockIdx.x * 32 + threadIdx.x], sgb[threadIdx.x]);
}

struct AdamC {
    float c1, c2, lr, decay, inv_batch;
};

__device__ __forceinline__ float adam1(const AdamC &c, float decay, float &p, float &m, float &v, float &gr) {
    const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    float g = gr * c.inv_batch;
    gr = 0.0f;
    m = b1 * m + (1.0f - b1) * g;
    v = b2 * v + (1.0f - b2) * g * g;
    float upd = (m / c.c1) / (sqrtf(v / c.c2) + eps);
    p = fminf(fmaxf(p * decay - c.lr * upd, -CLIP), CLIP);
    return p;
}

// Plain params (FT rows without the factoriser, or the rest of the net).
__global__ void adam_flat(AdamC c, float *p, float *m, float *v, float *g, float *wt, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float pp = p[i], mm = m[i], vv = v[i], gg = g[i];
    adam1(c, c.decay, pp, mm, vv, gg);
    p[i] = pp, m[i] = mm, v[i] = vv, g[i] = 0.0f;
    if (wt) wt[i] = pp;
}

// Factoriser: element e of F (768 x H): F from every bucket's gradient,
// then each bucket's W, then the merged rows into wt.
__global__ void adam_fac(AdamC c, float *pw, float *mw, float *vw, float *gw, float *pf, float *mf, float *vf,
                         float *wt, int H, int kb) {
    size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t nf = (size_t)INPUTS * H;
    if (e >= nf) return;
    float gf = 0;
    for (int b = 0; b < kb; b++) gf += gw[b * nf + e];
    float fp = pf[e], fm = mf[e], fv = vf[e];
    adam1(c, c.decay, fp, fm, fv, gf);
    pf[e] = fp, mf[e] = fm, vf[e] = fv;
    for (int b = 0; b < kb; b++) {
        size_t i = b * nf + e;
        float pp = pw[i], mm = mw[i], vv = vw[i], gg = gw[i];
        adam1(c, c.decay, pp, mm, vv, gg);
        pw[i] = pp, mw[i] = mm, vw[i] = vv, gw[i] = 0.0f;
        wt[i] = fminf(fmaxf(pp + fp, -CLIP), CLIP);
    }
}

// ---------------------------------------------------------------- main

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: bench <records.bin> [--hidden H] [--buckets B] [--kb none|kb8m] [--batches N] [--batch N]\n");
        return 1;
    }
    const char *path = argv[1];
    int H = 512, buckets = 8, nb = 200, B = 16384;
    std::string kbn = "kb8m";
    bool tiled = true, sorted_fwd = true;
    int WI = 2048;
    for (int i = 2; i + 1 < argc; i += 2) {
        std::string a = argv[i];
        if (a == "--hidden") H = atoi(argv[i + 1]);
        else if (a == "--buckets") buckets = atoi(argv[i + 1]);
        else if (a == "--kb") kbn = argv[i + 1];
        else if (a == "--batches") nb = atoi(argv[i + 1]);
        else if (a == "--batch") B = atoi(argv[i + 1]);
        else if (a == "--bwd") tiled = std::string(argv[i + 1]) == "tiled";
        else if (a == "--fwd-order") sorted_fwd = std::string(argv[i + 1]) == "bucket";
        else if (a == "--wi") WI = atoi(argv[i + 1]);
        else { fprintf(stderr, "unknown %s\n", a.c_str()); return 1; }
    }
    Layout kl = make_layout(kbn);
    int kb = kl.kb;
    bool fac = kb > 1;
    size_t rows = (size_t)kb * INPUTS, nft = rows * H, nrest = H + (size_t)buckets * 2 * H + buckets;

    // Data: the first nb batches, shuffled, random mirror flags (as --mirror).
    FILE *fp = fopen(path, "rb");
    if (!fp) { perror(path); return 1; }
    size_t nrec = (size_t)nb * B;
    std::vector<uint8_t> raw(nrec * 32);
    size_t got = fread(raw.data(), 32, nrec, fp);
    fclose(fp);
    if (got < nrec) { nb = got / B; nrec = (size_t)nb * B; }
    Rng rng{1 ^ 0x9E3779B97F4A7C15ull};
    std::vector<uint32_t> perm(nrec);
    for (size_t i = 0; i < nrec; i++) perm[i] = i;
    for (size_t i = nrec - 1; i > 0; i--) std::swap(perm[i], perm[rng.next() % (i + 1)]);
    std::vector<Sample> samples(nrec);
    auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < nrec; i++) samples[i] = decode(&raw[(size_t)perm[i] * 32], 0.5f, buckets, rng.next() & 1, kl);
    double dec_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    // Per batch: (sample, side) pairs stably sorted by king bucket, cut into
    // work items of up to WI pairs of one bucket.
    // Forward visit order: samples grouped by (stm, other) king bucket.
    std::vector<uint32_t> order(nrec);
    for (int bi = 0; bi < nb; bi++) {
        std::vector<int> cnt(kb * kb + 1, 0);
        const Sample *sb_ = &samples[(size_t)bi * B];
        auto key = [&](const Sample &s) { return s.f[0][0] / INPUTS * kb + s.f[1][0] / INPUTS; };
        for (int i = 0; i < B; i++) cnt[key(sb_[i]) + 1]++;
        for (int k = 0; k < kb * kb; k++) cnt[k + 1] += cnt[k];
        for (int i = 0; i < B; i++) order[(size_t)bi * B + cnt[key(sb_[i])]++] = i;
    }
    std::vector<uint32_t> pairs(nrec * 2);
    std::vector<Item> items;
    std::vector<int> item_start(nb + 1);
    for (int bi = 0; bi < nb; bi++) {
        item_start[bi] = items.size();
        std::vector<int> cnt(kb + 1, 0);
        const Sample *sb_ = &samples[(size_t)bi * B];
        auto kbk = [&](const Sample &s, int side) { return s.f[side][0] / INPUTS; };
        for (int i = 0; i < B; i++)
            for (int side = 0; side < 2; side++) cnt[kbk(sb_[i], side) + 1]++;
        for (int b = 0; b < kb; b++) cnt[b + 1] += cnt[b];
        std::vector<int> pos(cnt.begin(), cnt.end() - 1);
        uint32_t *out = &pairs[(size_t)bi * B * 2];
        for (int i = 0; i < B; i++)
            for (int side = 0; side < 2; side++) out[pos[kbk(sb_[i], side)]++] = (uint32_t)i << 1 | side;
        for (int b = 0; b < kb; b++)
            for (int a = cnt[b]; a < cnt[b + 1]; a += WI) items.push_back({b, (int)((size_t)bi * B * 2 + a), (int)((size_t)bi * B * 2 + std::min(a + WI, cnt[b + 1]))});
    }
    item_start[nb] = items.size();

    // Params (init_params / init_factor shapes; W = 0 with the factoriser).
    size_t nf = (size_t)INPUTS * H;
    std::vector<float> p_ft(nft, 0.0f), p_f(fac ? nf : 0), p_rest(nrest, 0.0f);
    Rng prng{5 ^ 0x9E3779B97F4A7C15ull};
    float a1 = 1.0f / sqrtf(32.0f), a2 = 1.0f / sqrtf(2.0f * H);
    if (fac) for (auto &x : p_f) x = prng.uniform(a1);
    else for (auto &x : p_ft) x = prng.uniform(a1);
    for (size_t i = H; i < H + (size_t)buckets * 2 * H; i++) p_rest[i] = prng.uniform(a2);
    std::vector<float> wt(nft);
    for (size_t i = 0; i < nft; i++) wt[i] = fac ? fminf(fmaxf(p_ft[i] + p_f[i % nf], -CLIP), CLIP) : p_ft[i];

    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("%s, H=%d B=%d %s%s, batch %d, %d batches, backward %s (decode %.2fs on 1 CPU thread)\n", prop.name, H,
           buckets, kbn.c_str(), fac ? " factorised" : "", B, nb, tiled ? "tiled" : "atomic", dec_s);

    auto dalloc = [](size_t n) { float *p; CK(cudaMalloc(&p, n * 4 + 16)); CK(cudaMemset(p, 0, n * 4 + 16)); return p; };
    float *d_pft = dalloc(nft), *d_mft = dalloc(nft), *d_vft = dalloc(nft), *d_gft = dalloc(nft), *d_wt = dalloc(nft);
    float *d_pf = dalloc(nf), *d_mf = dalloc(nf), *d_vf = dalloc(nf);
    float *d_prest = dalloc(nrest), *d_mrest = dalloc(nrest), *d_vrest = dalloc(nrest), *d_grest = dalloc(nrest);
    float *d_acc = dalloc((size_t)B * 2 * H), *d_gout = dalloc(B), *d_loss = dalloc(B);
    CK(cudaMemcpy(d_pft, p_ft.data(), nft * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_wt, wt.data(), nft * 4, cudaMemcpyHostToDevice));
    if (fac) CK(cudaMemcpy(d_pf, p_f.data(), nf * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_prest, p_rest.data(), nrest * 4, cudaMemcpyHostToDevice));
    Sample *d_s;
    CK(cudaMalloc(&d_s, nrec * sizeof(Sample)));
    CK(cudaMemcpy(d_s, samples.data(), nrec * sizeof(Sample), cudaMemcpyHostToDevice));
    uint32_t *d_order;
    CK(cudaMalloc(&d_order, nrec * 4));
    CK(cudaMemcpy(d_order, order.data(), nrec * 4, cudaMemcpyHostToDevice));
    uint32_t *d_pairs;
    Item *d_items;
    CK(cudaMalloc(&d_pairs, pairs.size() * 4));
    CK(cudaMemcpy(d_pairs, pairs.data(), pairs.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMalloc(&d_items, items.size() * sizeof(Item)));
    CK(cudaMemcpy(d_items, items.data(), items.size() * sizeof(Item), cudaMemcpyHostToDevice));

    const float *d_ftb = d_prest, *d_ow = d_prest + H, *d_ob = d_prest + H + (size_t)buckets * 2 * H;
    float *g_ftb = d_grest, *g_ow = d_grest + H, *g_ob = d_grest + H + (size_t)buckets * 2 * H;
    const int SPB = 32;
    float lr = 1e-3f;
    uint64_t step = 0;
    cudaEvent_t ev[5];
    for (auto &e : ev) CK(cudaEventCreate(&e));
    double tk[3] = {0, 0, 0};
    std::vector<float> hloss(B);

    auto run_batch = [&](int bi, bool timed) {
        const Sample *s = d_s + (size_t)bi * B;
        step++;
        AdamC c{1.0f - powf(0.9f, (float)step), 1.0f - powf(0.999f, (float)step), lr, 1.0f, 1.0f / B};
        if (timed) CK(cudaEventRecord(ev[0]));
        forward<<<B, H / 2>>>(s, sorted_fwd ? d_order + (size_t)bi * B : nullptr, d_wt, d_ftb, d_ow, d_ob, d_acc, d_gout,
                              d_loss, H);
        if (timed) CK(cudaEventRecord(ev[1]));
        if (tiled) {
            int ni = item_start[bi + 1] - item_start[bi];
            backward_tiled<<<dim3(H / 32, ni), 1024>>>(s, d_pairs, d_items + item_start[bi], d_acc, d_gout, d_ow, d_gft,
                                                      g_ftb, g_ow, g_ob, H, buckets);
        } else {
            backward<SPB><<<(B + SPB - 1) / SPB, H / 2>>>(s, d_acc, d_gout, d_ow, d_gft, g_ftb, g_ow, g_ob, H, B, buckets);
        }
        if (timed) CK(cudaEventRecord(ev[2]));
        if (fac)
            adam_fac<<<(nf + 255) / 256, 256>>>(c, d_pft, d_mft, d_vft, d_gft, d_pf, d_mf, d_vf, d_wt, H, kb);
        else
            adam_flat<<<(nft + 255) / 256, 256>>>(c, d_pft, d_mft, d_vft, d_gft, d_wt, nft);
        adam_flat<<<(nrest + 255) / 256, 256>>>(c, d_prest, d_mrest, d_vrest, d_grest, nullptr, nrest);
        if (timed) {
            CK(cudaEventRecord(ev[3]));
            CK(cudaEventSynchronize(ev[3]));
            float ms;
            for (int k = 0; k < 3; k++) {
                CK(cudaEventElapsedTime(&ms, ev[k], ev[k + 1]));
                tk[k] += ms;
            }
        }
    };

    // Warm-up, then the timed pass (no per-kernel events), then a profiled pass.
    for (int i = 0; i < 5; i++) run_batch(i % nb, false);
    CK(cudaDeviceSynchronize());
    double lsum = 0;
    int lcount = 0;
    auto tstart = std::chrono::steady_clock::now();
    for (int i = 0; i < nb; i++) {
        run_batch(i, false);
        if (i % 25 == 24 || i == nb - 1) {
            CK(cudaMemcpy(hloss.data(), d_loss, B * 4, cudaMemcpyDeviceToHost));
            double l = 0;
            for (float x : hloss) l += x;
            lsum += l / B, lcount++;
            printf("  batch %4d loss %.6f\n", i + 1, l / B);
        }
    }
    CK(cudaDeviceSynchronize());
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - tstart).count();
    printf("end to end (data on GPU): %.2f ms/batch, %.2fM pos/s\n", secs * 1e3 / nb, nb * (double)B / secs / 1e6);
    int np = std::min(nb, 50);
    for (int i = 0; i < np; i++) run_batch(i, true);
    printf("per kernel (avg of %d): forward %.2f ms, backward %.2f ms, adam %.2f ms\n", np, tk[0] / np, tk[1] / np,
           tk[2] / np);
    printf("host decode would need %.1fM pos/s per CPU thread (here %.1fM)\n", nb * (double)B / secs / 1e6,
           nrec / dec_s / 1e6);
    return 0;
}
