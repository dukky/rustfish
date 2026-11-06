import Foundation
// MSL source, compiled at runtime (no offline `metal` compiler needed).
//
// Parameter layout matches the CPU trainer / exported file:
//   ft weights [768][H] | ft bias [H] | out weights [B][2H] (stm half first) | out bias [B]
//
// Per batch: decode -> forward_train -> sort_entries -> grad_fused + grad_ob -> adam.
// Gradients are written as per-super-range partials and summed in a fixed
// order by `adam`. The only non-determinism is summation order inside
// grad_fused (bucket order from sort_entries, CAS flush order), i.e. float
// rounding, not semantics.

/// grad_fused column tile width, samples per sorted range, max output buckets.
let TW = Int(ProcessInfo.processInfo.environment["NNUE_TW"] ?? "8")!
let RS = Int(ProcessInfo.processInfo.environment["NNUE_RS"] ?? "128")!
let MAXB = 16

func kernelSource() -> String {
    "#include <metal_stdlib>\nusing namespace metal;\n#define TW \(TW)\n#define RS \(RS)\n#define MAXB \(MAXB)\n#define VARIANT \(ProcessInfo.processInfo.environment["NNUE_VARIANT"] ?? "0")\n" + kernelBody
}

let kernelBody = """
constant float SCALE = 400.0f;
constant float CLIP = 1.98f;

struct Cfg {
    uint h;
    uint nb;
    uint count;      // samples in this dispatch
    uint rf;         // grad_ft sample ranges
    float lambda;
    float lr;
    float decay;     // 1 - lr * wd
    float c1;
    float c2;
    float inv_batch;
    uint mode;       // adam: 0 = AdamW step, 1 = write summed gradient to m (gradcheck)
};

inline float sigm(float x) { return 1.0f / (1.0f + precise::exp(-x)); }

// 32-byte record -> features of both perspectives (0 = side to move),
// meta = n | bucket << 8, target. Byte 28 (padding) carries the mirror flag
// chosen on the CPU so the mirror RNG stream matches the CPU trainer.
kernel void decode(device const uint* recs [[buffer(0)]],
                   device ushort* feats [[buffer(1)]],
                   device uint* meta [[buffer(2)]],
                   device float* target [[buffer(3)]],
                   constant Cfg& c [[buffer(4)]],
                   uint s [[thread_position_in_grid]]) {
    if (s >= c.count) return;
    device const uint* r = recs + s * 8;
    uint lo = r[0], hi = r[1];
    uint w6 = r[6];
    float score = float(short(w6 & 0xFFFF));
    uint result = (w6 >> 16) & 0xFF;
    bool stm_black = ((w6 >> 24) & 0xFF) == 1;
    uint flip = (r[7] & 0xFF) != 0 ? 7 : 0;
    uint n = 0;
    for (uint half_ = 0; half_ < 2; half_++) {
        uint occ = half_ == 0 ? lo : hi;
        while (occ != 0 && n < 32) {
            uint sq = (ctz(occ) + half_ * 32) ^ flip;
            occ &= occ - 1;
            uint byte_ = (r[2 + (n >> 3)] >> (8 * ((n >> 1) & 3))) & 0xFF;
            uint nib = (byte_ >> (4 * (n & 1))) & 0xF;
            bool black = (nib >> 3) == 1;
            uint kind = nib & 7;
            for (uint i = 0; i < 2; i++) {
                bool pb = i == 0 ? stm_black : !stm_black;
                uint side = black == pb ? 0 : 384;
                uint rel = pb ? (sq ^ 56) : sq;
                feats[s * 64 + i * 32 + n] = ushort(side + kind * 64 + rel);
            }
            n++;
        }
    }
    uint bucket = c.nb == 1 ? 0 : min((n < 2 ? 0 : n - 2) * c.nb / 32, c.nb - 1);
    meta[s] = n | (bucket << 8);
    float wdl = float(result) / 2.0f;
    float sc = score;
    if (stm_black) { wdl = 1.0f - wdl; sc = -sc; }
    target[s] = c.lambda * wdl + (1.0f - c.lambda) * sigm(sc / SCALE);
}

// One threadgroup per sample; each thread owns float4 column chunks of both
// perspectives. Returns the net output (valid in thread 0).
inline float forward_one(device const float* p, device const ushort* feats, device const uint* meta,
                         device float* acc, bool store, constant Cfg& c, uint s,
                         uint tid, uint tpg, uint sl, uint sg, uint nsg, threadgroup float* red,
                         threadgroup ushort* fl) {
    uint h = c.h, h4 = c.h / 4;
    uint m = meta[s];
    uint n = m & 0xFF, b = m >> 8;
    for (uint i = tid; i < 64; i += tpg) fl[i] = feats[s * 64 + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const float4* w4 = (device const float4*)p;
    device const float4* fb4 = (device const float4*)(p + 768 * h);
    device const float4* ow4 = (device const float4*)(p + 769 * h + b * 2 * h);
    device float4* acc4 = (device float4*)(acc + s * 2 * h);
    float part = 0.0f;
    for (uint ch = tid; ch < 2 * h4; ch += tpg) {
        uint side = ch >= h4 ? 1 : 0;
        uint j4 = ch - side * h4;
        threadgroup const ushort* f = fl + side * 32;
        float4 x = fb4[j4];
        for (uint k = 0; k < n; k++) x += w4[uint(f[k]) * h4 + j4];
        if (store) acc4[ch] = x;
        float4 a = clamp(x, 0.0f, 1.0f);
        float4 ow = ow4[ch];
        // Same per-element order as the CPU: a * a * w.
        part += a.x * a.x * ow.x + a.y * a.y * ow.y + a.z * a.z * ow.z + a.w * a.w * ow.w;
    }
    float sum = simd_sum(part);
    if (sl == 0) red[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = 0.0f;
    if (tid == 0) {
        for (uint i = 0; i < nsg; i++) t += red[i];
    }
    return p[769 * h + c.nb * 2 * h + b] + t;
}

kernel void forward_train(device const float* p [[buffer(0)]],
                          device const ushort* feats [[buffer(1)]],
                          device const uint* meta [[buffer(2)]],
                          device const float* target [[buffer(3)]],
                          device float* acc [[buffer(4)]],
                          device float* gs [[buffer(5)]],
                          device float* loss [[buffer(6)]],
                          constant Cfg& c [[buffer(7)]],
                          uint s [[threadgroup_position_in_grid]],
                          uint tid [[thread_position_in_threadgroup]],
                          uint tpg [[threads_per_threadgroup]],
                          uint sl [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]],
                          uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float red[32];
    threadgroup ushort fl[64];
    float out = forward_one(p, feats, meta, acc, true, c, s, tid, tpg, sl, sg, nsg, red, fl);
    if (tid == 0) {
        float pred = sigm(out);
        float err = pred - target[s];
        gs[s] = 2.0f * err * pred * (1.0f - pred);
        loss[s] = err * err;
    }
}

kernel void forward_val(device const float* p [[buffer(0)]],
                        device const ushort* feats [[buffer(1)]],
                        device const uint* meta [[buffer(2)]],
                        device const float* target [[buffer(3)]],
                        device float* loss [[buffer(4)]],
                        device float* aeval [[buffer(5)]],
                        constant Cfg& c [[buffer(6)]],
                        uint s [[threadgroup_position_in_grid]],
                        uint tid [[thread_position_in_threadgroup]],
                        uint tpg [[threads_per_threadgroup]],
                        uint sl [[thread_index_in_simdgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float red[32];
    threadgroup ushort fl[64];
    float out = forward_one(p, feats, meta, (device float*)nullptr, false, c, s, tid, tpg, sl, sg, nsg, red, fl);
    if (tid == 0) {
        float d = sigm(out) - target[s];
        loss[s] = d * d;
        aeval[s] = fabs(out * SCALE);
    }
}

// Backward, step 1: per range of RS samples, counting-sort the active
// (feature, row) entries by feature. row = local sample * 2 + perspective.
// Output: sorted[range][RS*64] = feature << 16 | row, and the run offsets
// offs[range][769] (feature f occupies sorted[offs[f] .. offs[f+1])).
kernel void sort_entries(device const ushort* feats [[buffer(0)]],
                         device const uint* meta [[buffer(1)]],
                         device uint* sorted [[buffer(2)]],
                         device uint* offs [[buffer(3)]],
                         constant Cfg& c [[buffer(4)]],
                         uint rg [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint sl [[thread_index_in_simdgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup atomic_uint cnt[768];
    threadgroup uint simd_tot[8];
    for (uint i = tid; i < 768; i += 256) atomic_store_explicit(&cnt[i], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = tid; row < 2 * RS; row += 256) {
        uint s = rg * RS + row / 2;
        if (s < c.count) {
            uint n = meta[s] & 0xFF;
            device const ushort* f = feats + s * 64 + (row & 1) * 32;
            for (uint k = 0; k < n; k++) atomic_fetch_add_explicit(&cnt[f[k]], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Exclusive prefix sum over the 768 counts, 3 per thread.
    uint c0 = atomic_load_explicit(&cnt[3 * tid], memory_order_relaxed);
    uint c1 = atomic_load_explicit(&cnt[3 * tid + 1], memory_order_relaxed);
    uint c2 = atomic_load_explicit(&cnt[3 * tid + 2], memory_order_relaxed);
    uint local = c0 + c1 + c2;
    uint pre = simd_prefix_exclusive_sum(local);
    if (sl == 31) simd_tot[sg] = pre + local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint base = pre;
    for (uint i = 0; i < sg; i++) base += simd_tot[i];
    if (tid == 255) offs[rg * 769 + 768] = base + local;
    offs[rg * 769 + 3 * tid] = base;
    offs[rg * 769 + 3 * tid + 1] = base + c0;
    offs[rg * 769 + 3 * tid + 2] = base + c0 + c1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    atomic_store_explicit(&cnt[3 * tid], base, memory_order_relaxed);
    atomic_store_explicit(&cnt[3 * tid + 1], base + c0, memory_order_relaxed);
    atomic_store_explicit(&cnt[3 * tid + 2], base + c0 + c1, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device uint* out = sorted + rg * RS * 64;
    for (uint row = tid; row < 2 * RS; row += 256) {
        uint s = rg * RS + row / 2;
        if (s < c.count) {
            uint n = meta[s] & 0xFF;
            device const ushort* f = feats + s * 64 + (row & 1) * 32;
            for (uint k = 0; k < n; k++) {
                uint pos = atomic_fetch_add_explicit(&cnt[f[k]], 1u, memory_order_relaxed);
                out[pos] = (uint(f[k]) << 16) | row;
            }
        }
    }
}

// Float add on threadgroup memory via CAS (no native threadgroup float atomics).
inline void tg_add(threadgroup atomic_uint* a, float d) {
    uint old = atomic_load_explicit(a, memory_order_relaxed);
    while (!atomic_compare_exchange_weak_explicit(a, &old, as_type<uint>(as_type<float>(old) + d),
                                                  memory_order_relaxed, memory_order_relaxed)) {}
}

// Add a run's column sums into the tile: exclusive owner -> plain
// read-modify-write; a piece of a long run shared by several threads -> CAS.
inline void flush_run(threadgroup atomic_uint* tile, uint f, thread float* sum, bool shared) {
    for (uint q = 0; q < TW; q++) {
        threadgroup atomic_uint* a = &tile[f * TW + q];
#if VARIANT == 1
        if (sum[q] == 1234.5f) tg_add(a, 1.0f); sum[q] = 0.0f; continue;
#endif
        if (shared) {
            if (sum[q] != 0.0f) tg_add(a, sum[q]);
        } else {
            float o = as_type<float>(atomic_load_explicit(a, memory_order_relaxed));
            atomic_store_explicit(a, as_type<uint>(o + sum[q]), memory_order_relaxed);
        }
        sum[q] = 0.0f;
    }
}

// Backward, step 2. Threadgroup (tile, sr) owns columns [tile*TW, +TW) of
// both perspectives and the sample ranges of super-range sr. Per range it
// stages dacc[row][TW] in threadgroup memory (accumulating the output-weight
// and FT-bias gradients from the same acc values on the way), then walks the
// feature-sorted entry list in equal contiguous chunks per thread, summing
// runs of one feature in registers and flushing each run once into the
// threadgroup gradient tile [768][TW]. A run no longer than a chunk belongs
// entirely to the thread whose chunk holds its first entry (plain adds, no
// other thread touches that feature in this range); longer runs are split at
// chunk borders and their pieces added with CAS.
kernel void grad_fused(device const float* p [[buffer(0)]],
                       device const uint* meta [[buffer(1)]],
                       device const float* acc [[buffer(2)]],
                       device const float* gs [[buffer(3)]],
                       device const uint* sorted [[buffer(4)]],
                       device const uint* offs [[buffer(5)]],
                       device float* pft [[buffer(6)]],
                       device float* pow_ [[buffer(7)]],
                       constant Cfg& c [[buffer(8)]],
                       uint2 tg [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint tile[768 * TW];
    threadgroup float dt[2 * RS * TW];
    uint h = c.h;
    uint col = tid % TW;
    uint j = tg.x * TW + col;
    uint rowlane = tid / TW;              // this thread stages rows rowlane + k * (256 / TW)
    uint side = rowlane & 1;              // fixed per thread since 256 / TW is even
    for (uint i = tid; i < 768 * TW; i += 256) atomic_store_explicit(&tile[i], 0u, memory_order_relaxed);
    uint nranges = (c.count + RS - 1) / RS;
    uint per = (nranges + c.rf - 1) / c.rf;
    uint r0 = min(nranges, tg.y * per), r1 = min(nranges, r0 + per);
    device const float* ow = p + 769 * h;
    float bacc = 0.0f;
    float oacc[MAXB];
    for (uint b = 0; b < MAXB; b++) oacc[b] = 0.0f;
    float sum[TW];
    for (uint q = 0; q < TW; q++) sum[q] = 0.0f;
    for (uint rg = r0; rg < r1; rg++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint row = rowlane; row < 2 * RS; row += 256 / TW) {
            uint s = rg * RS + row / 2;
            float d = 0.0f;
            if (s < c.count) {
                float g = gs[s];
                uint b = meta[s] >> 8;
                float x = acc[s * 2 * h + side * h + j];
                float q = clamp(x, 0.0f, 1.0f);
                oacc[b] += g * q * q;
                if (x > 0.0f && x < 1.0f) d = g * ow[b * 2 * h + side * h + j] * 2.0f * x;
                bacc += d;
            }
            dt[row * TW + col] = d;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
#if VARIANT == 2
        continue;
#endif
        device const uint* off = offs + rg * 769;
        uint E = off[768];
        uint chunk = (E + 255) / 256;
        uint e0 = min(E, tid * chunk), e1 = min(E, e0 + chunk);
        device const uint* srt = sorted + rg * RS * 64;
        uint e = e0;
        while (e < e1) {
            uint f = srt[e] >> 16;
            uint a = off[f], b = off[f + 1];
            bool shared = b - a > chunk;
            if (a < e0 && !shared) { e = b; continue; }  // owned by an earlier thread
            uint stop = shared ? min(b, e1) : b;
            for (; e < stop; e++) {
                threadgroup const float4* d4 = (threadgroup const float4*)(dt + (srt[e] & 0xFFFF) * TW);
                for (uint q = 0; q < TW / 4; q++) {
                    float4 dv = d4[q];
                    sum[4 * q] += dv.x; sum[4 * q + 1] += dv.y; sum[4 * q + 2] += dv.z; sum[4 * q + 3] += dv.w;
                }
            }
            flush_run(tile, f, sum, shared);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float* out = pft + tg.y * 769 * h;
    for (uint i = tid; i < 768 * TW; i += 256) {
        out[(i / TW) * h + tg.x * TW + (i % TW)] = as_type<float>(atomic_load_explicit(&tile[i], memory_order_relaxed));
    }
    // FT bias: reduce the 256 / TW threads sharing each column, fixed order.
    dt[tid] = bacc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < TW) {
        float t = 0.0f;
        for (uint l = 0; l < 256 / TW; l++) t += dt[l * TW + tid];
        out[768 * h + tg.x * TW + tid] = t;
    }
    // Output weights: reduce the threads sharing (side, col), per bucket.
    for (uint b = 0; b < c.nb; b++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        dt[tid] = oacc[b];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 2 * TW) {
            uint sd = tid / TW, cl = tid % TW;
            float t = 0.0f;
            for (uint l = sd; l < 256 / TW; l += 2) t += dt[l * TW + cl];
            pow_[tg.y * c.nb * 2 * h + b * 2 * h + sd * h + tg.x * TW + cl] = t;
        }
    }
}

// Output bias gradient: thread (b, r) sums g over super-range r's samples in bucket b.
kernel void grad_ob(device const float* gs [[buffer(0)]],
                    device const uint* meta [[buffer(1)]],
                    device float* pob [[buffer(2)]],
                    constant Cfg& c [[buffer(3)]],
                    uint2 gid [[thread_position_in_grid]]) {
    uint b = gid.x, r = gid.y;
    if (b >= c.nb || r >= c.rf) return;
    uint nranges = (c.count + RS - 1) / RS;
    uint per = (nranges + c.rf - 1) / c.rf;
    uint s0 = min(c.count, r * per * RS), s1 = min(c.count, (r + 1) * per * RS);
    float t = 0.0f;
    for (uint s = s0; s < s1; s++) if ((meta[s] >> 8) == b) t += gs[s];
    pob[r * c.nb + b] = t;
}

// Sum partials in fixed order, then AdamW + clipping (same formula as the CPU trainer).
kernel void adam(device float* p [[buffer(0)]],
                 device float* m [[buffer(1)]],
                 device float* v [[buffer(2)]],
                 device const float* pft [[buffer(3)]],
                 device const float* pow_ [[buffer(4)]],
                 device const float* pob [[buffer(5)]],
                 constant Cfg& c [[buffer(6)]],
                 uint i [[thread_position_in_grid]]) {
    uint h = c.h;
    uint out_w = 769 * h, out_b = out_w + c.nb * 2 * h;
    if (i >= out_b + c.nb) return;
    float g = 0.0f;
    if (i < out_w) {
        for (uint r = 0; r < c.rf; r++) g += pft[r * out_w + i];
    } else if (i < out_b) {
        uint k = i - out_w;
        for (uint r = 0; r < c.rf; r++) g += pow_[r * c.nb * 2 * h + k];
    } else {
        uint k = i - out_b;
        for (uint r = 0; r < c.rf; r++) g += pob[r * c.nb + k];
    }
    if (c.mode == 1) { m[i] = g; return; }
    g *= c.inv_batch;
    const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    float mi = b1 * m[i] + (1.0f - b1) * g;
    float vi = b2 * v[i] + (1.0f - b2) * g * g;
    m[i] = mi;
    v[i] = vi;
    float upd = (mi / c.c1) / (sqrt(vi / c.c2) + eps);
    float x = p[i] * c.decay - c.lr * upd;
    p[i] = clamp(x, -CLIP, CLIP);
}
"""
