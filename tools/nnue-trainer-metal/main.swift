// Metal (GPU) NNUE trainer for rustfish: same nets, data, maths, RNG streams,
// export and log format as tools/nnue-trainer (v2), see README.md.

import Foundation
import Metal

let INPUTS = 768
let QA: Float = 255, QB: Float = 64, SCALE: Float = 400, CLIP: Float = 1.98
let REC = 32
let CKPT_MAGIC: [UInt8] = Array("RFNNT2\0\0".utf8)

func die(_ msg: String) -> Never {
    FileHandle.standardError.write((msg + "\n").data(using: .utf8)!)
    exit(1)
}

func out(_ s: String) {
    print(s)
    fflush(stdout)
}

// ---------------------------------------------------------------- RNG (bit-identical to the CPU trainer)

struct Rng {
    var s: UInt64
    init(_ seed: UInt64) { s = seed ^ 0x9E37_79B9_7F4A_7C15 }
    mutating func next() -> UInt64 {
        s = s &+ 0x9E37_79B9_7F4A_7C15
        var z = s
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    mutating func uniform(_ a: Float) -> Float {
        (Float(next() >> 40) / Float(UInt64(1) << 24) * 2.0 - 1.0) * a
    }
}

/// llvm.powi semantics (repeated squaring), as Rust's f32::powi.
func powi(_ a0: Float, _ n0: Int) -> Float {
    var a = a0, n = n0, r: Float = 1
    while true {
        if n & 1 == 1 { r *= a }
        n >>= 1
        if n == 0 { break }
        a *= a
    }
    return r
}

func sigmoid(_ x: Float) -> Float { 1.0 / (1.0 + expf(-x)) }

// ---------------------------------------------------------------- formatting (Rust-compatible)

func f6(_ x: Double) -> String { x.isNaN ? "NaN" : String(format: "%.6f", x) }
func f1(_ x: Double) -> String { x.isNaN ? "NaN" : String(format: "%.1f", x) }

/// Rust `{:.2e}`: "9.34e-4", "1.00e0".
func e2(_ x: Float) -> String {
    let s = String(format: "%.2e", Double(x))
    guard let r = s.range(of: "e") else { return s }
    return String(s[..<r.lowerBound]) + "e" + String(Int(s[r.upperBound...])!)
}

/// Rust `{}` for f32: shortest round-trip digits, never exponent notation.
func rustDisplay(_ x: Float) -> String {
    var d = x.description
    if d.hasSuffix(".0") { d.removeLast(2) }
    guard let er = d.range(of: "e") else { return d }
    var mant = String(d[..<er.lowerBound])
    let exp = Int(d[er.upperBound...])!
    var neg = false
    if mant.hasPrefix("-") { neg = true; mant.removeFirst() }
    let parts = mant.split(separator: ".", omittingEmptySubsequences: false)
    let ip = String(parts[0]), fp = parts.count > 1 ? String(parts[1]) : ""
    let digits = ip + fp
    let point = ip.count + exp
    var res: String
    if point <= 0 {
        res = "0." + String(repeating: "0", count: -point) + digits
    } else if point >= digits.count {
        res = digits + String(repeating: "0", count: point - digits.count)
    } else {
        let i = digits.index(digits.startIndex, offsetBy: point)
        res = String(digits[..<i]) + "." + String(digits[i...])
    }
    return (neg ? "-" : "") + res
}

// ---------------------------------------------------------------- shape / net I/O

struct Shape: Equatable {
    var h: Int
    var buckets: Int
    var ftB: Int { INPUTS * h }
    var outW: Int { ftB + h }
    var outB: Int { outW + buckets * 2 * h }
    var len: Int { outB + buckets }
    var fileBytes: Int { len * 2 }
    static func fromFileBytes(_ bytes: Int, _ buckets: Int) -> Shape? {
        let words = bytes / 2
        guard words >= buckets else { return nil }
        let h = (words - buckets) / (INPUTS + 1 + 2 * buckets)
        let s = Shape(h: h, buckets: buckets)
        return s.fileBytes == bytes ? s : nil
    }
    func scale(_ i: Int) -> Float { i < outW ? QA : (i < outB ? QB : QA * QB) }
}

func quantize(_ p: UnsafePointer<Float>, _ sh: Shape) -> [UInt8] {
    var outb = [UInt8](repeating: 0, count: sh.fileBytes)
    for i in 0..<sh.len {
        let q = Int16(min(max((p[i] * sh.scale(i)).rounded(.toNearestOrAwayFromZero), -32768), 32767))
        let u = UInt16(bitPattern: q)
        outb[2 * i] = UInt8(u & 0xFF)
        outb[2 * i + 1] = UInt8(u >> 8)
    }
    return outb
}

func dequantize(_ bytes: [UInt8], _ sh: Shape) -> [Float] {
    (0..<sh.len).map { i in
        Float(Int16(bitPattern: UInt16(bytes[2 * i]) | UInt16(bytes[2 * i + 1]) << 8)) / sh.scale(i)
    }
}

func readFile(_ path: String) -> [UInt8] {
    guard let d = FileManager.default.contents(atPath: path) else { die("\(path): cannot read") }
    return [UInt8](d)
}

func writeFile(_ path: String, _ bytes: [UInt8]) {
    guard FileManager.default.createFile(atPath: path, contents: Data(bytes)) else { die("\(path): cannot write") }
}

func loadNet(_ path: String, _ buckets: Int) -> ([Float], Shape) {
    let b = readFile(path)
    guard let sh = Shape.fromFileBytes(b.count, buckets) else {
        die("\(path): \(b.count) bytes is no (768->H)x2->\(buckets) net")
    }
    return (dequantize(b, sh), sh)
}

func initParams(_ sh: Shape, _ rng: inout Rng) -> [Float] {
    var p = [Float](repeating: 0, count: sh.len)
    var a: Float = 1.0 / Float(32).squareRoot()
    for i in 0..<(INPUTS * sh.h) { p[i] = rng.uniform(a) }
    a = 1.0 / Float(2 * sh.h).squareRoot()
    for i in sh.outW..<sh.outB { p[i] = rng.uniform(a) }
    return p
}

/// "out.bin" -> "out.e3.bin" (Rust Path::with_extension semantics for the common case).
func withExtension(_ path: String, _ ext: String) -> String {
    let url = URL(fileURLWithPath: path)
    let name = url.lastPathComponent
    let dir = (path as NSString).deletingLastPathComponent
    var stem = name
    if let dot = name.lastIndex(of: "."), dot != name.startIndex { stem = String(name[..<dot]) }
    let f = stem + "." + ext
    return dir.isEmpty ? f : dir + "/" + f
}

// ---------------------------------------------------------------- checkpoints (CPU trainer format)

func saveCkpt(_ path: String, _ sh: Shape, _ p: UnsafePointer<Float>, _ m: UnsafePointer<Float>, _ v: UnsafePointer<Float>, step: UInt64, seen: UInt64) {
    var d = Data(CKPT_MAGIC)
    for x in [UInt64(sh.h), UInt64(sh.buckets), step, seen] { withUnsafeBytes(of: x.littleEndian) { d.append(contentsOf: $0) } }
    for arr in [p, m, v] { d.append(UnsafeBufferPointer(start: arr, count: sh.len)) }
    FileManager.default.createFile(atPath: path, contents: d)
}

func loadCkpt(_ path: String, _ sh: Shape) -> ([Float], [Float], [Float], UInt64, UInt64) {
    let b = readFile(path)
    guard b.count >= 40, Array(b[0..<8]) == CKPT_MAGIC else { die("not a v2 checkpoint") }
    func word(_ i: Int) -> UInt64 { b[8 + 8 * i..<16 + 8 * i].reversed().reduce(0) { $0 << 8 | UInt64($1) } }
    guard Int(word(0)) == sh.h, Int(word(1)) == sh.buckets else { die("checkpoint shape differs") }
    let n = sh.len
    guard b.count == 40 + 12 * n else { die("checkpoint size mismatch") }
    var f = [Float](repeating: 0, count: 3 * n)
    b.withUnsafeBytes { raw in
        f.withUnsafeMutableBytes { $0.copyMemory(from: UnsafeRawBufferPointer(rebasing: raw[40...])) }
    }
    return (Array(f[0..<n]), Array(f[n..<2 * n]), Array(f[2 * n..<3 * n]), word(2), word(3))
}

// ---------------------------------------------------------------- data feed (same order as the CPU trainer)

/// Streams records from files in order, shuffling within 1M-record chunks.
final class Feed {
    static let CHUNK = 1 << 20
    let files: [String]
    var idx = 0
    var fp: UnsafeMutablePointer<FILE>? = nil
    var buf: UnsafeMutableRawPointer
    var count = 0
    var pos = 0
    var rng: Rng

    init(_ files: [String], seed: UInt64) {
        self.files = files
        rng = Rng(seed)
        buf = UnsafeMutableRawPointer.allocate(byteCount: Feed.CHUNK * REC, alignment: 16)
    }

    func refill() {
        count = 0
        pos = 0
        var filled = 0
        let cap = Feed.CHUNK * REC
        while filled < cap {
            if fp == nil {
                let f = files[idx % files.count]
                fp = fopen(f, "rb")
                if fp == nil { die("\(f): cannot open") }
            }
            let n = fread(buf + filled, 1, cap - filled, fp)
            if n == 0 {
                fclose(fp)
                fp = nil
                idx += 1
                if filled > 0 { break }
                continue
            }
            filled += n
        }
        filled -= filled % REC
        count = filled / REC
        // Fisher-Yates identical to the CPU trainer's `shuffle`.
        let w = buf.assumingMemoryBound(to: UInt64.self)
        var i = count - 1
        while i >= 1 {
            let j = Int(rng.next() % UInt64(i + 1))
            if i != j {
                for k in 0..<4 { let t = w[i * 4 + k]; w[i * 4 + k] = w[j * 4 + k]; w[j * 4 + k] = t }
            }
            i -= 1
        }
    }

    func batch(_ dst: UnsafeMutableRawPointer, _ n: Int) {
        var got = 0
        while got < n {
            if pos == count { refill() }
            let take = min(n - got, count - pos)
            memcpy(dst + got * REC, buf + pos * REC, take * REC)
            pos += take
            got += take
        }
    }
}

// ---------------------------------------------------------------- GPU context

struct Cfg {
    var h: UInt32 = 0
    var nb: UInt32 = 1
    var count: UInt32 = 0
    var rf: UInt32 = 8
    var lambda: Float = 0.5
    var lr: Float = 0
    var decay: Float = 1
    var c1: Float = 1
    var c2: Float = 1
    var invBatch: Float = 1
    var mode: UInt32 = 0
}

final class Gpu {
    let dev: MTLDevice
    let q: MTLCommandQueue
    let decode, fwdTrain, fwdVal, sortE, gradFused, gradOb, adam: MTLComputePipelineState

    init() {
        guard let d = MTLCreateSystemDefaultDevice() else { die("no Metal device") }
        dev = d
        q = d.makeCommandQueue()!
        let opts = MTLCompileOptions()
        if #available(macOS 15.0, *) { opts.mathMode = .safe } else { opts.fastMathEnabled = false }
        let lib: MTLLibrary
        do { lib = try d.makeLibrary(source: kernelSource(), options: opts) } catch { die("MSL compile failed: \(error)") }
        func pipe(_ n: String) -> MTLComputePipelineState {
            do { let ps = try d.makeComputePipelineState(function: lib.makeFunction(name: n)!); if ProcessInfo.processInfo.environment["NNUE_DEBUG"] != nil { FileHandle.standardError.write("\(n): tgmem \(ps.staticThreadgroupMemoryLength) maxthreads \(ps.maxTotalThreadsPerThreadgroup)\n".data(using: .utf8)!) }; return ps } catch { die("pipeline \(n): \(error)") }
        }
        decode = pipe("decode"); fwdTrain = pipe("forward_train"); fwdVal = pipe("forward_val")
        sortE = pipe("sort_entries"); gradFused = pipe("grad_fused"); gradOb = pipe("grad_ob"); adam = pipe("adam")
    }

    func buf(_ bytes: Int) -> MTLBuffer { dev.makeBuffer(length: max(bytes, 16), options: .storageModeShared)! }
    func buf(_ a: [Float]) -> MTLBuffer {
        let b = buf(a.count * 4)
        a.withUnsafeBytes { b.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        return b
    }
}

func fwdThreads(_ h: Int) -> Int { min(256, max(32, h / 2)) }

/// Decoded positions living on the GPU.
final class Batch {
    let n: Int
    let recs, feats, meta, target, loss, aux: MTLBuffer
    init(_ g: Gpu, _ n: Int) {
        self.n = n
        recs = g.buf(n * REC); feats = g.buf(n * 128); meta = g.buf(n * 4); target = g.buf(n * 4)
        loss = g.buf(n * 4); aux = g.buf(n * 4)
    }
    func encodeDecode(_ g: Gpu, _ enc: MTLComputeCommandEncoder, _ cfg: Cfg) {
        var c = cfg
        c.count = UInt32(n)
        enc.setComputePipelineState(g.decode)
        enc.setBuffers([recs, feats, meta, target], offsets: [0, 0, 0, 0], range: 0..<4)
        enc.setBytes(&c, length: MemoryLayout<Cfg>.stride, index: 4)
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    func sum(_ b: MTLBuffer) -> Double {
        let p = b.contents().assumingMemoryBound(to: Float.self)
        var s = 0.0
        for i in 0..<n { s += Double(p[i]) }
        return s
    }
}

/// Training state: params, Adam moments, scratch for one batch.
final class Trainer {
    let g: Gpu
    let sh: Shape
    let batchN: Int
    var cfg: Cfg
    let p, m, v, acc, gs, pft, pow_, pob, sorted, ecount: MTLBuffer

    init(_ g: Gpu, _ sh: Shape, batch: Int, lambda: Float, params: [Float], m0: [Float]?, v0: [Float]?) {
        self.g = g
        self.sh = sh
        batchN = batch
        precondition(sh.h % TW == 0, "--hidden must be a multiple of \(TW)")
        precondition(sh.buckets <= MAXB, "at most \(MAXB) buckets")
        cfg = Cfg()
        cfg.h = UInt32(sh.h); cfg.nb = UInt32(sh.buckets); cfg.lambda = lambda
        cfg.rf = UInt32(max(1, min(8, (batch + RS - 1) / RS)))
        p = g.buf(params)
        m = m0.map { g.buf($0) } ?? g.buf(sh.len * 4)
        v = v0.map { g.buf($0) } ?? g.buf(sh.len * 4)
        if m0 == nil { memset(m.contents(), 0, sh.len * 4); memset(v.contents(), 0, sh.len * 4) }
        acc = g.buf(batch * 2 * sh.h * 4)
        gs = g.buf(batch * 4)
        pft = g.buf(Int(cfg.rf) * sh.outW * 4)
        pow_ = g.buf(Int(cfg.rf) * sh.buckets * 2 * sh.h * 4)
        pob = g.buf(Int(cfg.rf) * sh.buckets * 4)
        let nr = (batch + RS - 1) / RS
        sorted = g.buf(nr * RS * 64 * 4)
        ecount = g.buf(nr * 769 * 4)
    }

    /// Encode forward + backward + (mode 0) AdamW step / (mode 1) gradient into `m`.
    /// `only` restricts encoding to one stage (profiling).
    func encodeStep(_ cb: MTLCommandBuffer, _ b: Batch, lr: Float, wd: Float, c1: Float, c2: Float, mode: UInt32 = 0, only: Int = -1) {
        var c = cfg
        c.count = UInt32(b.n)
        c.lr = lr; c.decay = 1.0 - lr * wd; c.c1 = c1; c.c2 = c2
        c.invBatch = 1.0 / Float(b.n); c.mode = mode
        let cl = MemoryLayout<Cfg>.stride
        let enc = cb.makeComputeCommandEncoder()!
        if only < 0 || only == 0 { b.encodeDecode(g, enc, c) }
        enc.memoryBarrier(scope: .buffers)
        if only < 0 || only == 1 {
        enc.setComputePipelineState(g.fwdTrain)
        enc.setBuffers([p, b.feats, b.meta, b.target, acc, gs, b.loss], offsets: [0, 0, 0, 0, 0, 0, 0], range: 0..<7)
        enc.setBytes(&c, length: cl, index: 7)
        enc.dispatchThreadgroups(MTLSize(width: b.n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: fwdThreads(sh.h), height: 1, depth: 1))
        }
        enc.memoryBarrier(scope: .buffers)
        if only < 0 || only == 2 {
        enc.setComputePipelineState(g.sortE)
        enc.setBuffers([b.feats, b.meta, sorted, ecount], offsets: [0, 0, 0, 0], range: 0..<4)
        enc.setBytes(&c, length: cl, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: (b.n + RS - 1) / RS, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        enc.memoryBarrier(scope: .buffers)
        if only < 0 || only == 3 {
        enc.setComputePipelineState(g.gradFused)
        enc.setBuffers([p, b.meta, acc, gs, sorted, ecount, pft, pow_], offsets: [0, 0, 0, 0, 0, 0, 0, 0], range: 0..<8)
        enc.setBytes(&c, length: cl, index: 8)
        enc.dispatchThreadgroups(MTLSize(width: sh.h / TW, height: Int(c.rf), depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.setComputePipelineState(g.gradOb)
        enc.setBuffers([gs, b.meta, pob], offsets: [0, 0, 0], range: 0..<3)
        enc.setBytes(&c, length: cl, index: 3)
        enc.dispatchThreads(MTLSize(width: sh.buckets, height: Int(c.rf), depth: 1), threadsPerThreadgroup: MTLSize(width: sh.buckets, height: 1, depth: 1))
        }
        enc.memoryBarrier(scope: .buffers)
        if only < 0 || only == 4 {
        enc.setComputePipelineState(g.adam)
        enc.setBuffers([p, m, v, pft, pow_, pob], offsets: [0, 0, 0, 0, 0, 0], range: 0..<6)
        enc.setBytes(&c, length: cl, index: 6)
        enc.dispatchThreads(MTLSize(width: sh.len, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        enc.endEncoding()
    }
}

/// Validation set decoded once on the GPU; forward pass for any param buffer.
final class ValSet {
    let b: Batch?
    init(_ g: Gpu, _ recs: [UInt8], lambda: Float, sh: Shape) {
        let n = recs.count / REC
        guard n > 0 else { b = nil; return }
        let bb = Batch(g, n)
        recs.withUnsafeBytes { bb.recs.contents().copyMemory(from: $0.baseAddress!, byteCount: n * REC) }
        // Byte 28 = mirror flag: never mirror validation positions.
        let r = bb.recs.contents().assumingMemoryBound(to: UInt8.self)
        for i in 0..<n { r[i * REC + 28] = 0 }
        var c = Cfg()
        c.h = UInt32(sh.h); c.nb = UInt32(sh.buckets); c.lambda = lambda
        let cb = g.q.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        bb.encodeDecode(g, enc, c)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        b = bb
    }

    func validate(_ g: Gpu, _ p: MTLBuffer, _ sh: Shape) -> (Double, Double) {
        guard let b = b else { return (.nan, .nan) }
        var c = Cfg()
        c.h = UInt32(sh.h); c.nb = UInt32(sh.buckets); c.count = UInt32(b.n)
        let cb = g.q.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(g.fwdVal)
        enc.setBuffers([p, b.feats, b.meta, b.target, b.loss, b.aux], offsets: [0, 0, 0, 0, 0, 0], range: 0..<6)
        enc.setBytes(&c, length: MemoryLayout<Cfg>.stride, index: 6)
        enc.dispatchThreadgroups(MTLSize(width: b.n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: fwdThreads(sh.h), height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        return (b.sum(b.loss) / Double(b.n), b.sum(b.aux) / Double(b.n))
    }
}

// ---------------------------------------------------------------- argument parsing

final class Args {
    var a: [String]
    var i = 0
    init(_ a: [String]) { self.a = a }
    func next() -> String? { if i < a.count { i += 1; return a[i - 1] }; return nil }
    func value(_ flag: String) -> String { guard let v = next() else { die("missing value for \(flag)") }; return v }
}

func parse<T: LosslessStringConvertible>(_ s: String, _ flag: String) -> T {
    guard let v = T(s) else { die("bad value for \(flag): \(s)") }
    return v
}

// ---------------------------------------------------------------- commands

func cmdTrain(_ args: [String]) {
    var data: [String] = []
    var valPath: String? = nil
    var outPath = "net.bin"
    var sh = Shape(h: 256, buckets: 1)
    var epochs: UInt64 = 10, epochSize: UInt64 = 0
    var batch = 16384
    var lr: Float = 1e-3, lrEnd: Float = 1e-5, wd: Float = 0, lambda: Float = 0.5
    var threads = 8
    var seed: UInt64 = 1
    var mirror = false
    var initPath: String? = nil, resume: String? = nil, ckptPath: String? = nil
    let it = Args(args)
    while let a = it.next() {
        switch a {
        case "--data": data.append(it.value(a))
        case "--val": valPath = it.value(a)
        case "--out": outPath = it.value(a)
        case "--hidden": sh.h = parse(it.value(a), a)
        case "--buckets": sh.buckets = parse(it.value(a), a)
        case "--epochs": epochs = parse(it.value(a), a)
        case "--epoch-size": epochSize = parse(it.value(a), a)
        case "--batch": batch = parse(it.value(a), a)
        case "--lr": lr = parse(it.value(a), a)
        case "--lr-end": lrEnd = parse(it.value(a), a)
        case "--wd": wd = parse(it.value(a), a)
        case "--lambda": lambda = parse(it.value(a), a)
        case "--threads": threads = parse(it.value(a), a)  // accepted for CLI compatibility; unused
        case "--seed": seed = parse(it.value(a), a)
        case "--mirror": mirror = true
        case "--init": initPath = it.value(a)
        case "--resume": resume = it.value(a)
        case "--checkpoint": ckptPath = it.value(a)
        default: die("unknown option \(a)")
        }
    }
    if data.isEmpty { die("--data required") }
    if epochSize == 0 {
        epochSize = data.reduce(0) { s, p in
            let sz = (try? FileManager.default.attributesOfItem(atPath: p)[.size] as? UInt64) ?? nil
            guard let z = sz else { die("\(p): cannot stat") }
            return s + z / UInt64(REC)
        }
    }
    let total = epochs * epochSize
    let ckpt = ckptPath ?? withExtension(outPath, "ckpt")

    var rng = Rng(seed)
    var p0: [Float], m0: [Float]? = nil, v0: [Float]? = nil
    var step: UInt64 = 0, seen: UInt64 = 0
    if let r = resume {
        let (p, m, v, st, se) = loadCkpt(r, sh)
        p0 = p; m0 = m; v0 = v; step = st; seen = se
    } else if let i = initPath {
        let (p, s) = loadNet(i, sh.buckets)
        if s != sh { die("init net shape differs from --hidden/--buckets") }
        p0 = p.map { min(max($0, -CLIP), CLIP) }
    } else {
        p0 = initParams(sh, &rng)
    }
    let g = Gpu()
    let valRecs = valPath.map(readFile) ?? []
    let val = ValSet(g, valRecs, lambda: lambda, sh: sh)
    let tr = Trainer(g, sh, batch: batch, lambda: lambda, params: p0, m0: m0, v0: v0)
    out("train H=\(sh.h) buckets=\(sh.buckets) params=\(sh.len) epochs=\(epochs) x \(epochSize) batch=\(batch) lr \(rustDisplay(lr))->\(rustDisplay(lrEnd)) wd=\(rustDisplay(wd)) lambda=\(rustDisplay(lambda)) mirror=\(mirror) threads=\(threads) val=\(valRecs.count / REC)")
    let (vl0, ve0) = val.validate(g, tr.p, sh)
    out("start val loss \(f6(vl0)) mean|eval| \(f1(ve0))cp")

    // Loader thread: fills ring slots with raw records (+ mirror flag in byte 28),
    // drawing seed_b from the main RNG exactly as the CPU trainer does per batch.
    let slots = 3
    let ring = (0..<slots).map { _ in Batch(g, batch) }
    let free = DispatchSemaphore(value: slots), filled = DispatchSemaphore(value: 0)
    let feed = Feed(data, seed: seed ^ 0xF00D ^ seen)
    var loaderRng = rng
    let loader = Thread {
        var k = 0
        while true {
            free.wait()
            let b = ring[k % slots]
            let dst = b.recs.contents()
            feed.batch(dst, batch)
            let seedB = loaderRng.next()
            let r = dst.assumingMemoryBound(to: UInt8.self)
            let nChunks = (batch + 255) / 256
            for c in 0..<nChunks {
                var mr = Rng(seedB ^ UInt64(c))
                for i in (c * 256)..<min(batch, (c + 1) * 256) {
                    r[i * REC + 28] = (mirror && mr.next() & 1 == 1) ? 1 : 0
                }
            }
            filled.signal()
            k += 1
        }
    }
    loader.stackSize = 1 << 20
    loader.start()

    let started = Date()
    let startSeen = seen
    var epochLoss = 0.0
    var epochN: UInt64 = 0
    var ep = seen / epochSize
    var pending: [(MTLCommandBuffer, Batch)] = []
    var k = 0
    func retire() {
        let (cb, b) = pending.removeFirst()
        cb.waitUntilCompleted()
        if let e = cb.error { die("GPU error: \(e)") }
        epochLoss += b.sum(b.loss)
        free.signal()
    }
    let pP = tr.p.contents().assumingMemoryBound(to: Float.self)
    while seen < total {
        let t = Double(seen) / Double(total)
        let curLr = lrEnd + 0.5 * (lr - lrEnd) * (1.0 + Float(cos(Double.pi * t)))
        filled.wait()
        let b = ring[k % slots]
        k += 1
        step += 1
        let e = Int(min(step, 1 << 20))
        let c1 = 1.0 - powi(0.9, e), c2 = 1.0 - powi(0.999, e)
        let cb = g.q.makeCommandBuffer()!
        tr.encodeStep(cb, b, lr: curLr, wd: wd, c1: c1, c2: c2)
        cb.commit()
        pending.append((cb, b))
        if pending.count >= 2 { retire() }
        seen += UInt64(batch)
        epochN += UInt64(batch)

        if seen / epochSize != (seen - UInt64(batch)) / epochSize || seen >= total {
            while !pending.isEmpty { retire() }
            ep += 1
            let secs = Date().timeIntervalSince(started)
            let (vl, ve) = val.validate(g, tr.p, sh)
            let q = quantize(pP, sh)
            let qb = g.buf(dequantize(q, sh))
            let (ql, _) = val.validate(g, qb, sh)
            let path = withExtension(outPath, "e\(ep).bin")
            writeFile(path, q)
            saveCkpt(ckpt, sh, pP, tr.m.contents().assumingMemoryBound(to: Float.self), tr.v.contents().assumingMemoryBound(to: Float.self), step: step, seen: seen)
            out("epoch \(ep) lr \(e2(curLr)) train \(f6(epochLoss / Double(epochN))) val \(f6(vl)) (quantized \(f6(ql))) mean|eval| \(f1(ve))cp \(String(format: "%.0f", Double(seen - startSeen) / secs)) pos/s -> \(path)")
            epochLoss = 0
            epochN = 0
        }
    }
    while !pending.isEmpty { retire() }
    let q = quantize(pP, sh)
    writeFile(outPath, q)
    out("wrote \(outPath) (\(q.count) bytes, H=\(sh.h) buckets=\(sh.buckets))")
    exit(0)
}

func cmdEval(_ args: [String]) {
    var net: String? = nil, valPath: String? = nil
    var buckets = 1
    var lambda: Float = 0.5
    let it = Args(args)
    while let a = it.next() {
        switch a {
        case "--net": net = it.value(a)
        case "--val": valPath = it.value(a)
        case "--buckets": buckets = parse(it.value(a), a)
        case "--lambda": lambda = parse(it.value(a), a)
        default: die("unknown option \(a)")
        }
    }
    guard let n = net, let vp = valPath else { die("--net and --val required") }
    let (p, sh) = loadNet(n, buckets)
    let g = Gpu()
    let recs = readFile(vp)
    let val = ValSet(g, recs, lambda: lambda, sh: sh)
    let (l, e) = val.validate(g, g.buf(p), sh)
    out("H=\(sh.h) buckets=\(sh.buckets) val positions \(recs.count / REC) loss \(f6(l)) mean|eval| \(f1(e))cp")
}

// ---------------------------------------------------------------- CPU reference (gradcheck)

struct Sample { var f: [[Int]]; var bucket: Int; var target: Float }

func cpuDecode(_ r: UnsafePointer<UInt8>, _ lambda: Float, _ buckets: Int, _ mirror: Bool) -> Sample {
    var occ: UInt64 = 0
    for i in 0..<8 { occ |= UInt64(r[i]) << (8 * UInt64(i)) }
    let score = Float(Int16(bitPattern: UInt16(r[24]) | UInt16(r[25]) << 8))
    let stmBlack = r[27] == 1
    var s = Sample(f: [[], []], bucket: 0, target: 0)
    let flip = mirror ? 7 : 0
    var n = 0
    while occ != 0 {
        let sq = occ.trailingZeroBitCount ^ flip
        occ &= occ - 1
        let nib = Int(r[8 + n / 2] >> (4 * UInt8(n & 1)) & 0xF)
        let black = nib >> 3 == 1
        let kind = nib & 7
        for (i, pb) in [stmBlack, !stmBlack].enumerated() {
            s.f[i].append((black == pb ? 0 : 384) + kind * 64 + (pb ? sq ^ 56 : sq))
        }
        n += 1
    }
    s.bucket = buckets == 1 ? 0 : min(max(n - 2, 0) * buckets / 32, buckets - 1)
    var wdl = Float(r[26]) / 2.0
    var sc = score
    if stmBlack { wdl = 1.0 - wdl; sc = -sc }
    s.target = lambda * wdl + (1.0 - lambda) * sigmoid(sc / SCALE)
    return s
}

func cpuForward(_ p: [Float], _ sh: Shape, _ s: Sample, _ acc: inout [Float]) -> Float {
    let h = sh.h
    for side in 0..<2 {
        for j in 0..<h { acc[side * h + j] = p[sh.ftB + j] }
        for f in s.f[side] { for j in 0..<h { acc[side * h + j] += p[f * h + j] } }
    }
    let ow = sh.outW + s.bucket * 2 * h
    var sum: Float = 0
    for i in 0..<2 * h { let a = min(max(acc[i], 0), 1); sum += a * a * p[ow + i] }
    return p[sh.outB + s.bucket] + sum
}

func cpuTrainSample(_ p: [Float], _ sh: Shape, _ s: Sample, _ g: inout [Double]) -> Double {
    let h = sh.h
    var acc = [Float](repeating: 0, count: 2 * h)
    let o = cpuForward(p, sh, s, &acc)
    let pred = sigmoid(o)
    let err = pred - s.target
    let grad = 2.0 * err * pred * (1.0 - pred)
    let ow0 = sh.outW + s.bucket * 2 * h
    g[sh.outB + s.bucket] += Double(grad)
    var dacc = [Float](repeating: 0, count: 2 * h)
    for i in 0..<2 * h {
        let x = acc[i], a = min(max(x, 0), 1)
        g[ow0 + i] += Double(grad * a * a)
        dacc[i] = (x > 0 && x < 1) ? grad * p[ow0 + i] * 2.0 * a : 0
    }
    for side in 0..<2 {
        for j in 0..<h { g[sh.ftB + j] += Double(dacc[side * h + j]) }
        for f in s.f[side] { for j in 0..<h { g[f * h + j] += Double(dacc[side * h + j]) } }
    }
    return Double(err * err)
}

/// GPU gradient vs CPU reference on a small batch, finite differences on a few
/// parameters, and one AdamW step compared parameter by parameter.
func cmdGradcheck(_ args: [String]) {
    var dataPath: String? = nil
    var sh = Shape(h: 32, buckets: 8)
    var n = 256
    var mirror = false
    let it = Args(args)
    while let a = it.next() {
        switch a {
        case "--data": dataPath = it.value(a)
        case "--hidden": sh.h = parse(it.value(a), a)
        case "--buckets": sh.buckets = parse(it.value(a), a)
        case "--batch": n = parse(it.value(a), a)
        case "--mirror": mirror = true
        default: die("unknown option \(a)")
        }
    }
    guard let dp = dataPath else { die("--data required") }
    let lambda: Float = 0.5
    var raw = Array(readFile(dp).prefix(n * REC))
    n = raw.count / REC
    for i in 0..<n { raw[i * REC + 28] = mirror ? UInt8(i & 1) : 0 }
    var rng = Rng(7)
    var p = initParams(sh, &rng)
    // Bias the FT so many accumulators sit inside (0, 1) and all branches are exercised.
    for j in 0..<sh.h { p[sh.ftB + j] = 0.3 + 0.2 * rng.uniform(1) }
    for b in 0..<sh.buckets { p[sh.outB + b] = 0.05 * rng.uniform(1) }

    let g = Gpu()
    let tr = Trainer(g, sh, batch: n, lambda: lambda, params: p, m0: nil, v0: nil)
    let b = Batch(g, n)
    raw.withUnsafeBytes { b.recs.contents().copyMemory(from: $0.baseAddress!, byteCount: n * REC) }
    var cb = g.q.makeCommandBuffer()!
    tr.encodeStep(cb, b, lr: 0, wd: 0, c1: 1, c2: 1, mode: 1)
    cb.commit(); cb.waitUntilCompleted(); if let e = cb.error { die("GPU error: \(e)") }
    let gg = tr.m.contents().assumingMemoryBound(to: Float.self)

    var ref = [Double](repeating: 0, count: sh.len)
    var samples: [Sample] = []
    var lossRef = 0.0
    raw.withUnsafeBufferPointer { rp in
        for i in 0..<n {
            let s = cpuDecode(rp.baseAddress! + i * REC, lambda, sh.buckets, raw[i * REC + 28] != 0)
            samples.append(s)
            lossRef += cpuTrainSample(p, sh, s, &ref)
        }
    }
    let lossGpu = b.sum(b.loss)
    var maxAbs = 0.0, maxRef = 0.0, worst = 0
    for i in 0..<sh.len {
        let d = abs(Double(gg[i]) - ref[i])
        if d > maxAbs { maxAbs = d; worst = i }
        maxRef = max(maxRef, abs(ref[i]))
    }
    out("gradcheck H=\(sh.h) buckets=\(sh.buckets) batch=\(n) mirror=\(mirror)")
    out("  loss sum: gpu \(lossGpu) cpu \(lossRef) rel \(abs(lossGpu - lossRef) / lossRef)")
    out("  grad: max|gpu-cpu| \(maxAbs) at \(worst) (max|grad| \(maxRef)), rel \(maxAbs / maxRef)")

    // Finite differences on the summed loss (double precision accumulation).
    func lossAt(_ q: [Double]) -> Double {
        let h = sh.h
        var l = 0.0
        var acc = [Double](repeating: 0, count: 2 * h)
        for s in samples {
            for side in 0..<2 {
                for j in 0..<h { acc[side * h + j] = q[sh.ftB + j] }
                for f in s.f[side] { for j in 0..<h { acc[side * h + j] += q[f * h + j] } }
            }
            var o = q[sh.outB + s.bucket]
            for i in 0..<2 * h { let a = min(max(acc[i], 0), 1); o += a * a * q[sh.outW + s.bucket * 2 * h + i] }
            let d = 1.0 / (1.0 + exp(-o)) - Double(s.target)
            l += d * d
        }
        return l
    }
    let pd = p.map { Double($0) }
    var picks: [Int] = []
    let f0 = samples[0].f[0][0] * sh.h, f1i = samples[0].f[1][3] * sh.h
    picks += [f0, f0 + 3, f1i + 1, sh.ftB + 1, sh.ftB + sh.h - 1]
    picks += [sh.outW + samples[0].bucket * 2 * sh.h + 5, sh.outW + samples[0].bucket * 2 * sh.h + sh.h + 2, sh.outB + samples[0].bucket]
    var fdBad = 0
    for i in picks {
        let eps = 1e-6
        var a = pd; a[i] += eps
        var c = pd; c[i] -= eps
        let num = (lossAt(a) - lossAt(c)) / (2 * eps)
        let ok = abs(num - Double(gg[i])) < 1e-5 + 1e-3 * abs(num)
        if !ok { fdBad += 1 }
        out("  fd param \(i): numeric \(String(format: "%.6e", num)) gpu \(String(format: "%.6e", Double(gg[i]))) cpu \(String(format: "%.6e", ref[i])) \(ok ? "ok" : "MISMATCH")")
    }

    // One AdamW step (lr 1e-3, wd 0.01) vs CPU reference formula on the CPU reference gradient.
    let tr2 = Trainer(g, sh, batch: n, lambda: lambda, params: p, m0: nil, v0: nil)
    cb = g.q.makeCommandBuffer()!
    let lr: Float = 1e-3, wd: Float = 0.01
    let c1 = 1.0 - powi(0.9, 1), c2 = 1.0 - powi(0.999, 1)
    tr2.encodeStep(cb, b, lr: lr, wd: wd, c1: c1, c2: c2)
    cb.commit(); cb.waitUntilCompleted(); if let e = cb.error { die("GPU error: \(e)") }
    let p2 = tr2.p.contents().assumingMemoryBound(to: Float.self)
    var maxStep = 0.0
    var flips = 0
    for i in 0..<sh.len {
        let gr = Float(ref[i]) * (1.0 / Float(n))
        let m = (1.0 - Float(0.9)) * gr, v = (1.0 - Float(0.999)) * gr * gr
        let upd = (m / c1) / ((v / c2).squareRoot() + 1e-8)
        let x = min(max(p[i] * (1.0 - lr * wd) - lr * upd, -CLIP), CLIP)
        let d = abs(Double(p2[i]) - Double(x))
        // Adam normalises: a near-zero gradient can flip sign between summation orders.
        if d > 1e-6 { flips += 1 }
        maxStep = max(maxStep, d)
    }
    out("  adam step: max|gpu-cpu| param \(maxStep), params differing > 1e-6: \(flips) of \(sh.len)")
    let pass = maxAbs / maxRef < 1e-4 && fdBad == 0 && abs(lossGpu - lossRef) / lossRef < 1e-5
    out(pass ? "PASS" : "FAIL")
    exit(pass ? 0 : 1)
}

/// Per-stage GPU time for one batch (no data loading).
func cmdBench(_ args: [String]) {
    var dataPath: String? = nil
    var sh = Shape(h: 256, buckets: 1)
    var n = 16384, iters = 20
    let it = Args(args)
    while let a = it.next() {
        switch a {
        case "--data": dataPath = it.value(a)
        case "--hidden": sh.h = parse(it.value(a), a)
        case "--buckets": sh.buckets = parse(it.value(a), a)
        case "--batch": n = parse(it.value(a), a)
        case "--iters": iters = parse(it.value(a), a)
        default: die("unknown option \(a)")
        }
    }
    guard let dp = dataPath else { die("--data required") }
    var rng = Rng(1)
    let p = initParams(sh, &rng)
    let g = Gpu()
    let tr = Trainer(g, sh, batch: n, lambda: 0.5, params: p, m0: nil, v0: nil)
    let b = Batch(g, n)
    let raw = Array(readFile(dp).prefix(n * REC))
    raw.withUnsafeBytes { b.recs.contents().copyMemory(from: $0.baseAddress!, byteCount: n * REC) }
    let names = ["decode", "forward", "sort", "grad", "adam"]
    var tot = 0.0
    var line = "bench H=\(sh.h) B=\(sh.buckets) batch=\(n):"
    for st in -1..<5 {
        // Sustained: iters command buffers back to back (warm clocks), GPU span / iters.
        var best = 1e9
        for _ in 0..<3 {
            var cbs: [MTLCommandBuffer] = []
            for _ in 0..<iters {
                let cb = g.q.makeCommandBuffer()!
                tr.encodeStep(cb, b, lr: 1e-4, wd: 0, c1: 1, c2: 1, only: st)
                cb.commit()
                cbs.append(cb)
            }
            cbs.last!.waitUntilCompleted()
            if let e = cbs.last!.error { die("GPU error: \(e)") }
            best = min(best, (cbs.last!.gpuEndTime - cbs.first!.gpuStartTime) * 1000 / Double(iters))
        }
        if st < 0 { line += String(format: " step %.2fms (%.2fM pos/s) |", best, Double(n) / best / 1000); continue }
        tot += best
        line += String(format: " %@ %.2f", names[st], best)
    }
    out(line + String(format: " | sum %.2fms", tot))
}

let argv = Array(CommandLine.arguments.dropFirst())
let usage = "usage: nnue-trainer-metal <train|eval|gradcheck> ... (see README.md)"
guard let cmd = argv.first else { die(usage) }
let rest = Array(argv.dropFirst())
switch cmd {
case "train": cmdTrain(rest)
case "eval": cmdEval(rest)
case "gradcheck": cmdGradcheck(rest)
case "bench": cmdBench(rest)
default: die(usage)
}
