import Foundation
// MSL source, compiled at runtime (no offline `metal` compiler needed).
//
// Parameter layout matches the CPU trainer / exported file:
//   ft weights [KB*768][H] | ft bias [H] | out weights [B][2H] (stm half first) | out bias [B]
// (NF = KB*768 FT rows; KB = king buckets, see `--king-buckets`).
//
// Per batch: decode -> forward_train -> plan_rows -> sort_entries -> grad_fused + grad_ob
// (+ row_active) -> adam (+ merge_factor).
// Gradients are written as per-work-item partials and summed in a fixed
// order by `adam`; every reduction has a fixed order, so training is
// deterministic run to run (it differs from the CPU trainer only in float
// summation order).

/// grad_fused column tile width (= SIMD width), samples per sorted range
/// (2 * RS = 256 rows = threads per sort group), max output buckets.
let TW = 32
let RS = 128
let MAXB = 16
/// Max king buckets (FT rows = KB * 768).
let MAXKB = 16
/// grad_fused simdgroups per threadgroup (768 / NSG gradient registers per lane).
let NSG = Int(ProcessInfo.processInfo.environment["NNUE_NSG"] ?? "32")!

func kernelSource() -> String {
    "#include <metal_stdlib>\nusing namespace metal;\n#define TW \(TW)\n#define RS \(RS)\n#define MAXB \(MAXB)\n#define MAXKB \(MAXKB)\n#define NSG \(NSG)\n" + kernelBody
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
    uint nf;         // FT rows = KB * 768
    uint kmirror;    // king layout mirrors (king on files e-h -> flip files)
    uint plen;       // net params (the exported layout)
    uint fl;         // factoriser floats after them (768 * H), 0 = off
    uint wdact;      // weight decay only on FT rows active in the batch
    uint ktab[16];   // king bucket per perspective-relative square, 4 per word
};

inline uint king_bucket(constant Cfg& c, uint k) { return (c.ktab[k >> 2] >> (8 * (k & 3))) & 0xFF; }

inline float sigm(float x) { return 1.0f / (1.0f + precise::exp(-x)); }

// 32-byte record -> features of both perspectives (0 = side to move),
// meta = n | bucket << 8 | king bucket (persp 0) << 16 | king bucket (persp 1) << 24, target. Byte 28 (padding) carries the mirror flag
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
    // King contexts (0 = side to move) from each perspective's own king.
    uint kbase[2] = {0, 0}, kflip[2] = {0, 0};
    if (c.nf > 768 || c.kmirror != 0) {
        uint ksq[2] = {0, 0};
        uint n = 0;
        for (uint half_ = 0; half_ < 2; half_++) {
            uint occ = half_ == 0 ? lo : hi;
            while (occ != 0 && n < 32) {
                uint sq = (ctz(occ) + half_ * 32) ^ flip;
                occ &= occ - 1;
                uint byte_ = (r[2 + (n >> 3)] >> (8 * ((n >> 1) & 3))) & 0xFF;
                uint nib = (byte_ >> (4 * (n & 1))) & 0xF;
                if ((nib & 7) == 5) ksq[nib >> 3] = sq;
                n++;
            }
        }
        for (uint i = 0; i < 2; i++) {
            bool pb = i == 0 ? stm_black : !stm_black;
            uint k = pb ? (ksq[1] ^ 56) : ksq[0];
            kbase[i] = king_bucket(c, k) * 768;
            kflip[i] = (c.kmirror != 0 && (k & 7) >= 4) ? 7 : 0;
        }
    }
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
                feats[s * 64 + i * 32 + n] = ushort(kbase[i] + side + kind * 64 + (rel ^ kflip[i]));
            }
            n++;
        }
    }
    uint bucket = c.nb == 1 ? 0 : min((n < 2 ? 0 : n - 2) * c.nb / 32, c.nb - 1);
    meta[s] = n | (bucket << 8) | ((kbase[0] / 768) << 16) | ((kbase[1] / 768) << 24);
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
    uint n = m & 0xFF, b = (m >> 8) & 0xFF;
    for (uint i = tid; i < 64; i += tpg) fl[i] = feats[s * 64 + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const float4* w4 = (device const float4*)p;
    device const float4* fb4 = (device const float4*)(p + c.nf * h);
    device const float4* ow4 = (device const float4*)(p + (c.nf + 1) * h + b * 2 * h);
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
    return p[(c.nf + 1) * h + c.nb * 2 * h + b] + t;
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

// Backward, step 0 (one threadgroup): order the batch's rows (row = sample *
// 2 + perspective) by king bucket, keeping increasing row order inside each
// bucket (stable, deterministic). Every row's features lie in its king's
// 768-feature block, so a bucket's rows form one dense 768-feature problem.
// Each bucket's list is cut into ranges of up to 2 * RS rows, and the
// ranges into work items of up to `per` ranges (per = the one-bucket
// super-range length). With one king bucket the order is the identity and
// ranges and items are exactly the old per-128-sample ranges and
// super-ranges, so KB = 1 sums in the same order as before.
// hdr = [nranges, nitems, first item per bucket (kb + 1 entries)],
// rng[r] = (kb, first row slot, rows), item[i] = (kb, first range, end range).
kernel void plan_rows(device const uint* meta [[buffer(0)]],
                      device uint* rows [[buffer(1)]],
                      device uint* hdr [[buffer(2)]],
                      device uint4* rng [[buffer(3)]],
                      device uint4* item [[buffer(4)]],
                      constant Cfg& c [[buffer(5)]],
                      uint tid [[thread_position_in_threadgroup]],
                      uint tpg [[threads_per_threadgroup]],
                      uint sl [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint simd_tot[32];
    threadgroup uint tot[MAXKB];
    uint nkb = c.nf / 768;
    uint nrows = 2 * c.count;
    uint chunk = (nrows + tpg - 1) / tpg;
    uint r0 = min(nrows, tid * chunk), r1 = min(nrows, r0 + chunk);
    uint cnt[MAXKB];
    for (uint k = 0; k < MAXKB; k++) cnt[k] = 0;
    for (uint r = r0; r < r1; r++) cnt[(meta[r >> 1] >> (16 + 8 * (r & 1))) & 0xFF]++;
    // Per bucket: exclusive prefix of the thread counts (thread order = row order).
    for (uint k = 0; k < nkb; k++) {
        uint pre = simd_prefix_exclusive_sum(cnt[k]);
        if (sl == 31) simd_tot[sg] = pre + cnt[k];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint base = pre;
        for (uint i = 0; i < sg; i++) base += simd_tot[i];
        if (tid == tpg - 1) tot[k] = base + cnt[k];
        cnt[k] = base;   // now this thread's first slot inside bucket k
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint gst[MAXKB];
    uint t = 0;
    for (uint k = 0; k < nkb; k++) { gst[k] = t; t += tot[k]; }
    for (uint r = r0; r < r1; r++) {
        uint k = (meta[r >> 1] >> (16 + 8 * (r & 1))) & 0xFF;
        rows[gst[k] + cnt[k]++] = r;
    }
    if (tid != 0) return;
    uint nr1 = (c.count + RS - 1) / RS;
    uint per = (nr1 + c.rf - 1) / c.rf;
    uint nr = 0, ni = 0;
    for (uint k = 0; k < nkb; k++) {
        hdr[2 + k] = ni;
        uint n = tot[k], first = nr;
        for (uint o = 0; o < n; o += 2 * RS) rng[nr++] = uint4(k, gst[k] + o, min(2u * RS, n - o), 0);
        for (uint a = first; a < nr; a += per) item[ni++] = uint4(k, a, min(nr, a + per), 0);
    }
    hdr[2 + nkb] = ni;
    hdr[0] = nr;
    hdr[1] = ni;
}

// Backward, step 1: per range (up to 256 rows of one king bucket kb, local
// row = position in the range), bucket the active (feature, row) entries by
// feature, deterministically: a 256-bit row mask per feature (atomic OR is
// order independent), then each feature's rows are emitted in increasing
// order. Output: sorted[range][RS*64] = local row, offs[range][769] (local
// feature f = global - kb*768 occupies sorted[offs[f] .. offs[f+1])).
kernel void sort_entries(device const ushort* feats [[buffer(0)]],
                         device const uint* meta [[buffer(1)]],
                         device ushort* sorted [[buffer(2)]],
                         device uint* offs [[buffer(3)]],
                         device const uint* rows [[buffer(4)]],
                         device const uint* hdr [[buffer(5)]],
                         device const uint4* rng [[buffer(6)]],
                         constant Cfg& c [[buffer(7)]],
                         uint tgp [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint sl [[thread_index_in_simdgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup atomic_uint mask[768 * 8];
    threadgroup uint simd_tot[8];
    uint rg = tgp;
    if (rg >= hdr[0]) return;   // uniform per threadgroup
    uint4 R = rng[rg];
    uint kb = R.x;
    for (uint i = tid; i < 768 * 8; i += 256) atomic_store_explicit(&mask[i], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
        uint row = tid;
        if (row < R.z) {
            uint rid = rows[R.y + row];
            uint s = rid >> 1;
            uint n = meta[s] & 0xFF;
            device const ushort* f = feats + s * 64 + (rid & 1) * 32;
            for (uint k = 0; k < n; k++)
                atomic_fetch_or_explicit(&mask[(uint(f[k]) - kb * 768) * 8 + row / 32], 1u << (row % 32), memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint cnt[3];
    for (uint i = 0; i < 3; i++) {
        uint t = 0;
        for (uint w = 0; w < 8; w++) t += popcount(atomic_load_explicit(&mask[(3 * tid + i) * 8 + w], memory_order_relaxed));
        cnt[i] = t;
    }
    uint local = cnt[0] + cnt[1] + cnt[2];
    uint pre = simd_prefix_exclusive_sum(local);
    if (sl == 31) simd_tot[sg] = pre + local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint base = pre;
    for (uint i = 0; i < sg; i++) base += simd_tot[i];
    device uint* off = offs + rg * 769;
    if (tid == 255) off[768] = base + local;
    device ushort* out = sorted + rg * RS * 64;
    uint pos = base;
    for (uint i = 0; i < 3; i++) {
        off[3 * tid + i] = pos;
        for (uint w = 0; w < 8; w++) {
            uint bits = atomic_load_explicit(&mask[(3 * tid + i) * 8 + w], memory_order_relaxed);
            while (bits != 0) {
                out[pos++] = ushort(w * 32 + ctz(bits));
                bits &= bits - 1;
            }
        }
    }
}

// Backward, step 2. Threadgroup (tile, work item): NSG simdgroups x 32
// lanes; lane = one of 32 columns [tile*32, +32). The item is a run of
// ranges of one king bucket kb. For each range it stages dacc[row][32] in
// threadgroup memory (accumulating the FT-bias and output-weight gradients
// from the same acc values), then simdgroup k sums, for each feature
// f = k, k+NSG, ... of kb's block, the staged rows of f's run into a
// register: uniform control flow, one broadcast index load and one
// conflict-free threadgroup load per entry, no atomics. At the end the
// per-(feature, column) sums go to this item's own slot of the
// partial-gradient buffer (768 FT rows + FT bias), so the result is
// deterministic. Every row is staged by exactly one item, so the work is
// the same as for one king bucket (plus at most one partial range per bucket).
// ONE (one king bucket): rows are in identity order, so range rows need no
// lookup and simdgroup sg only ever stages side sg & 1 (one output-gradient
// register set instead of two).
template <bool ONE>
inline void grad_body(device const float* p, device const uint* meta, device const float* acc,
                      device const float* gs, device const ushort* sorted, device const uint* offs,
                      device float* pft, device float* pow_, device const uint* rows,
                      device const uint* hdr, device const uint4* rng, device const uint4* item,
                      constant Cfg& c, uint2 tg, uint lane, uint sg, threadgroup float* dt) {
    if (tg.y >= hdr[1]) return;   // uniform per threadgroup
    uint h = c.h;
    uint j = tg.x * 32 + lane;
    bool live = j < h;
    uint4 I = item[tg.y];
    device const float* ow = p + (c.nf + 1) * h;
    device float* G = pft + tg.y * 769 * h;
    float bacc = 0.0f;
    float oacc[ONE ? MAXB : 2 * MAXB];    // [side][output bucket]
    for (uint b = 0; b < (ONE ? MAXB : 2 * MAXB); b++) oacc[b] = 0.0f;
    float gr[768 / NSG];                  // this lane's sums for features sg + NSG * i
    for (uint i = 0; i < 768 / NSG; i++) gr[i] = 0.0f;
    for (uint rg = I.y; rg < I.z; rg++) {
        uint4 R = rng[rg];
        device const uint* off = offs + rg * 769;
        device const ushort* srt = sorted + rg * RS * 64;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint row = sg; row < 2 * RS; row += NSG) {
            float d = 0.0f;
            if (row < R.z && live) {     // uniform per simdgroup
                uint rid = ONE ? R.y + row : rows[R.y + row];
                uint s = rid >> 1, side = ONE ? (sg & 1) : (rid & 1);
                uint ms = meta[s];
                float g = gs[s];
                uint b = (ms >> 8) & 0xFF;
                float x = acc[s * 2 * h + side * h + j];
                float q = clamp(x, 0.0f, 1.0f);
                oacc[ONE ? b : side * MAXB + b] += g * q * q;
                if (x > 0.0f && x < 1.0f) d = g * ow[b * 2 * h + side * h + j] * 2.0f * x;
                bacc += d;
            }
            dt[row * 32 + lane] = d;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
        for (uint i = 0; i < 768 / NSG; i++) {
            uint f = sg + NSG * i;
            uint a = off[f], b = off[f + 1];
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            uint e = a;
            for (; e + 4 <= b; e += 4) {
                s0 += dt[uint(srt[e]) * 32 + lane];
                s1 += dt[uint(srt[e + 1]) * 32 + lane];
                s2 += dt[uint(srt[e + 2]) * 32 + lane];
                s3 += dt[uint(srt[e + 3]) * 32 + lane];
            }
            for (; e < b; e++) s0 += dt[uint(srt[e]) * 32 + lane];
            gr[i] += (s0 + s1) + (s2 + s3);
        }
    }
    if (live) {
        for (uint i = 0; i < 768 / NSG; i++) G[(sg + NSG * i) * h + j] = gr[i];
    }
    // FT bias: reduce the NSG simdgroups per column, fixed order.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    dt[sg * 32 + lane] = bacc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && live) {
        float t = 0.0f;
        for (uint k = 0; k < NSG; k++) t += dt[k * 32 + lane];
        G[768 * h + j] = t;
    }
    // Output weights: per bucket, reduce the NSG simdgroups per (side, column)
    // in simdgroup order.
    if (ONE) {
        for (uint b = 0; b < c.nb; b++) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            dt[sg * 32 + lane] = oacc[b];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg < 2 && live) {
                float t = 0.0f;
                for (uint k = sg; k < NSG; k += 2) t += dt[k * 32 + lane];
                pow_[tg.y * c.nb * 2 * h + b * 2 * h + sg * h + j] = t;
            }
        }
        return;
    }
    for (uint b = 0; b < c.nb; b++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        dt[sg * 64 + lane] = oacc[b];
        dt[sg * 64 + 32 + lane] = oacc[ONE ? b : MAXB + b];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg < 2 && live) {
            float t = 0.0f;
            for (uint k = 0; k < NSG; k++) t += dt[k * 64 + sg * 32 + lane];
            pow_[tg.y * c.nb * 2 * h + b * 2 * h + sg * h + j] = t;
        }
    }
}

kernel void grad_fused(device const float* p [[buffer(0)]], device const uint* meta [[buffer(1)]],
                       device const float* acc [[buffer(2)]], device const float* gs [[buffer(3)]],
                       device const ushort* sorted [[buffer(4)]], device const uint* offs [[buffer(5)]],
                       device float* pft [[buffer(6)]], device float* pow_ [[buffer(7)]],
                       device const uint* rows [[buffer(8)]], device const uint* hdr [[buffer(9)]],
                       device const uint4* rng [[buffer(10)]], device const uint4* item [[buffer(11)]],
                       constant Cfg& c [[buffer(12)]], uint2 tg [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float dt[2 * RS * 32];
    grad_body<false>(p, meta, acc, gs, sorted, offs, pft, pow_, rows, hdr, rng, item, c, tg, lane, sg, dt);
}

kernel void grad_fused1(device const float* p [[buffer(0)]], device const uint* meta [[buffer(1)]],
                       device const float* acc [[buffer(2)]], device const float* gs [[buffer(3)]],
                       device const ushort* sorted [[buffer(4)]], device const uint* offs [[buffer(5)]],
                       device float* pft [[buffer(6)]], device float* pow_ [[buffer(7)]],
                       device const uint* rows [[buffer(8)]], device const uint* hdr [[buffer(9)]],
                       device const uint4* rng [[buffer(10)]], device const uint4* item [[buffer(11)]],
                       constant Cfg& c [[buffer(12)]], uint2 tg [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float dt[2 * RS * 32];
    grad_body<true>(p, meta, acc, gs, sorted, offs, pft, pow_, rows, hdr, rng, item, c, tg, lane, sg, dt);
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
    for (uint s = s0; s < s1; s++) if (((meta[s] >> 8) & 0xFF) == b) t += gs[s];
    pob[r * c.nb + b] = t;
}

// --wd-active-only: act[row] = 1 if FT row `row` (king bucket row / 768,
// feature row % 768) has an entry in any of that bucket's ranges.
kernel void row_active(device const uint* offs [[buffer(0)]],
                       device const uint4* rng [[buffer(1)]],
                       device const uint* hdr [[buffer(2)]],
                       device uint* act [[buffer(3)]],
                       constant Cfg& c [[buffer(4)]],
                       uint row [[thread_position_in_grid]]) {
    if (row >= c.nf) return;
    uint kb = row / 768, f = row % 768, nr = hdr[0];
    uint a = 0;
    for (uint rg = 0; rg < nr && a == 0; rg++) {
        if (rng[rg].x != kb) continue;
        device const uint* off = offs + rg * 769;
        a = off[f + 1] > off[f] ? 1 : 0;
    }
    act[row] = a;
}

// Sum partials in fixed order (FT rows: the items of the row's king bucket;
// FT bias, output weights: every item; output bias: every super-range;
// factoriser F: every item's 768 local feature rows, i.e. the sum over king
// buckets), then AdamW + clipping (same formula as the CPU trainer). `p` is
// the training vector: the net's params, then (factoriser) F.
kernel void adam(device float* p [[buffer(0)]],
                 device float* m [[buffer(1)]],
                 device float* v [[buffer(2)]],
                 device const float* pft [[buffer(3)]],
                 device const float* pow_ [[buffer(4)]],
                 device const float* pob [[buffer(5)]],
                 device const uint* hdr [[buffer(6)]],
                 constant Cfg& c [[buffer(7)]],
                 device const uint* act [[buffer(8)]],
                 uint i [[thread_position_in_grid]]) {
    uint h = c.h;
    uint ft = c.nf * h, out_w = ft + h, out_b = out_w + c.nb * 2 * h;
    if (i >= c.plen + c.fl) return;
    float g = 0.0f;
    if (i < ft) {
        uint kb = i / (768 * h), k = i - kb * 768 * h;
        uint i1 = hdr[3 + kb];
        for (uint r = hdr[2 + kb]; r < i1; r++) g += pft[r * 769 * h + k];
    } else if (i < out_w) {
        uint k = i - ft, ni = hdr[1];
        for (uint r = 0; r < ni; r++) g += pft[r * 769 * h + 768 * h + k];
    } else if (i < out_b) {
        uint k = i - out_w, ni = hdr[1];
        for (uint r = 0; r < ni; r++) g += pow_[r * c.nb * 2 * h + k];
    } else if (i < c.plen) {
        uint k = i - out_b;
        for (uint r = 0; r < c.rf; r++) g += pob[r * c.nb + k];
    } else {
        uint k = i - c.plen, ni = hdr[1];
        for (uint r = 0; r < ni; r++) g += pft[r * 769 * h + k];
    }
    if (c.mode == 1) { m[i] = g; return; }
    g *= c.inv_batch;
    const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    float mi = b1 * m[i] + (1.0f - b1) * g;
    float vi = b2 * v[i] + (1.0f - b2) * g * g;
    m[i] = mi;
    v[i] = vi;
    float upd = (mi / c.c1) / (sqrt(vi / c.c2) + eps);
    float decay = (c.wdact != 0 && i < ft && act[i / h] == 0) ? 1.0f : c.decay;
    float x = p[i] * decay - c.lr * upd;
    p[i] = clamp(x, -CLIP, CLIP);
}

// Factoriser: effective net params from the training vector `t`
// (FT row element = clamp(W + F), the rest copied).
kernel void merge_factor(device const float* t [[buffer(0)]],
                         device float* p [[buffer(1)]],
                         constant Cfg& c [[buffer(2)]],
                         uint i [[thread_position_in_grid]]) {
    if (i >= c.plen) return;
    p[i] = i < c.nf * c.h ? clamp(t[i] + t[c.plen + i % c.fl], -CLIP, CLIP) : t[i];
}
"""
