//! Minimal SIMD layer for the training kernels in `batch.rs`.
//!
//! Only lane-wise IEEE add/mul, min/max clamping and a mask select are used
//! (no FMA, no reassociation), so every implementation gives bit-identical
//! results; the kernels decide the summation order, never the vector width.
//! x86-64 picks AVX-512 / AVX2 at runtime, so one binary runs everywhere;
//! aarch64 uses NEON. `NNUE_ISA=portable|avx2|avx512|neon` overrides the
//! choice (for tests and benchmarks).

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Isa {
    Portable,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Avx512,
    #[cfg(target_arch = "aarch64")]
    Neon,
}

impl Isa {
    pub fn detect() -> Isa {
        let best = Self::best();
        match std::env::var("NNUE_ISA").ok().as_deref() {
            None | Some("") => best,
            Some("portable") => Isa::Portable,
            #[cfg(target_arch = "x86_64")]
            Some("avx2") if std::is_x86_feature_detected!("avx2") => Isa::Avx2,
            #[cfg(target_arch = "x86_64")]
            Some("avx512") if best == Isa::Avx512 => Isa::Avx512,
            #[cfg(target_arch = "aarch64")]
            Some("neon") => Isa::Neon,
            Some(o) => panic!("NNUE_ISA={o} is not available on this CPU"),
        }
    }

    fn best() -> Isa {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
                return Isa::Avx512;
            }
            if std::is_x86_feature_detected!("avx2") {
                return Isa::Avx2;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            return Isa::Neon;
        }
        #[allow(unreachable_code)]
        Isa::Portable
    }

    pub fn name(self) -> &'static str {
        match self {
            Isa::Portable => "portable",
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 => "avx2",
            #[cfg(target_arch = "x86_64")]
            Isa::Avx512 => "avx512",
            #[cfg(target_arch = "aarch64")]
            Isa::Neon => "neon",
        }
    }
}

/// A vector of `W` f32 lanes. All methods are unsafe: pointers must be valid
/// for `W` floats, and the CPU must support the implementation's ISA.
pub trait Simd: Copy {
    const W: usize;
    unsafe fn zero() -> Self;
    unsafe fn splat(x: f32) -> Self;
    unsafe fn load(p: *const f32) -> Self;
    unsafe fn store(self, p: *mut f32);
    unsafe fn add(self, o: Self) -> Self;
    unsafe fn mul(self, o: Self) -> Self;
    /// `x.clamp(0, 1)` (for non-NaN x; may differ in the sign of a zero,
    /// which no caller can observe: only `a * a` and masked uses follow).
    unsafe fn clamp01(self) -> Self;
    /// Lane-wise `if 0 < x < 1 { v } else { +0 }`.
    unsafe fn select_in01(x: Self, v: Self) -> Self;
}

/// Plain Rust lanes; LLVM vectorises these for whatever the build targets.
#[derive(Clone, Copy)]
pub struct Portable([f32; 8]);

impl Simd for Portable {
    const W: usize = 8;
    #[inline(always)]
    unsafe fn zero() -> Self {
        Portable([0.0; 8])
    }
    #[inline(always)]
    unsafe fn splat(x: f32) -> Self {
        Portable([x; 8])
    }
    #[inline(always)]
    unsafe fn load(p: *const f32) -> Self {
        Portable(std::ptr::read_unaligned(p as *const [f32; 8]))
    }
    #[inline(always)]
    unsafe fn store(self, p: *mut f32) {
        std::ptr::write_unaligned(p as *mut [f32; 8], self.0)
    }
    #[inline(always)]
    unsafe fn add(self, o: Self) -> Self {
        let mut r = self.0;
        for i in 0..8 {
            r[i] += o.0[i];
        }
        Portable(r)
    }
    #[inline(always)]
    unsafe fn mul(self, o: Self) -> Self {
        let mut r = self.0;
        for i in 0..8 {
            r[i] *= o.0[i];
        }
        Portable(r)
    }
    #[inline(always)]
    unsafe fn clamp01(self) -> Self {
        let mut r = self.0;
        for x in &mut r {
            *x = x.clamp(0.0, 1.0);
        }
        Portable(r)
    }
    #[inline(always)]
    unsafe fn select_in01(x: Self, v: Self) -> Self {
        let mut r = v.0;
        for i in 0..8 {
            if !(x.0[i] > 0.0 && x.0[i] < 1.0) {
                r[i] = 0.0;
            }
        }
        Portable(r)
    }
}

#[cfg(target_arch = "x86_64")]
pub use x86::{Avx2, Avx512};

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::Simd;
    use std::arch::x86_64::*;

    #[derive(Clone, Copy)]
    pub struct Avx2(__m256);

    impl Simd for Avx2 {
        const W: usize = 8;
        #[inline]
        #[target_feature(enable = "avx2,fma")]
        unsafe fn zero() -> Self {
            Avx2(_mm256_setzero_ps())
        }
        #[inline]
        #[target_feature(enable = "avx2,fma")]
        unsafe fn splat(x: f32) -> Self {
            Avx2(_mm256_set1_ps(x))
        }
        #[inline]
        #[target_feature(enable = "avx2,fma")]
        unsafe fn load(p: *const f32) -> Self {
            Avx2(_mm256_loadu_ps(p))
        }
        #[inline]
        #[target_feature(enable = "avx2,fma")]
        unsafe fn store(self, p: *mut f32) {
            _mm256_storeu_ps(p, self.0)
        }
        #[inline]
        #[target_feature(enable = "avx2,fma")]
        unsafe fn add(self, o: Self) -> Self {
            Avx2(_mm256_add_ps(self.0, o.0))
        }
        #[inline]
        #[target_feature(enable = "avx2,fma")]
        unsafe fn mul(self, o: Self) -> Self {
            Avx2(_mm256_mul_ps(self.0, o.0))
        }
        #[inline]
        #[target_feature(enable = "avx2,fma")]
        unsafe fn clamp01(self) -> Self {
            Avx2(_mm256_min_ps(_mm256_max_ps(self.0, _mm256_setzero_ps()), _mm256_set1_ps(1.0)))
        }
        #[inline]
        #[target_feature(enable = "avx2,fma")]
        unsafe fn select_in01(x: Self, v: Self) -> Self {
            let gt = _mm256_cmp_ps::<_CMP_GT_OQ>(x.0, _mm256_setzero_ps());
            let lt = _mm256_cmp_ps::<_CMP_LT_OQ>(x.0, _mm256_set1_ps(1.0));
            Avx2(_mm256_and_ps(_mm256_and_ps(gt, lt), v.0))
        }
    }

    #[derive(Clone, Copy)]
    pub struct Avx512(__m512);

    impl Simd for Avx512 {
        const W: usize = 16;
        #[inline]
        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,fma")]
        unsafe fn zero() -> Self {
            Avx512(_mm512_setzero_ps())
        }
        #[inline]
        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,fma")]
        unsafe fn splat(x: f32) -> Self {
            Avx512(_mm512_set1_ps(x))
        }
        #[inline]
        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,fma")]
        unsafe fn load(p: *const f32) -> Self {
            Avx512(_mm512_loadu_ps(p))
        }
        #[inline]
        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,fma")]
        unsafe fn store(self, p: *mut f32) {
            _mm512_storeu_ps(p, self.0)
        }
        #[inline]
        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,fma")]
        unsafe fn add(self, o: Self) -> Self {
            Avx512(_mm512_add_ps(self.0, o.0))
        }
        #[inline]
        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,fma")]
        unsafe fn mul(self, o: Self) -> Self {
            Avx512(_mm512_mul_ps(self.0, o.0))
        }
        #[inline]
        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,fma")]
        unsafe fn clamp01(self) -> Self {
            Avx512(_mm512_min_ps(_mm512_max_ps(self.0, _mm512_setzero_ps()), _mm512_set1_ps(1.0)))
        }
        #[inline]
        #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx2,fma")]
        unsafe fn select_in01(x: Self, v: Self) -> Self {
            let gt = _mm512_cmp_ps_mask::<_CMP_GT_OQ>(x.0, _mm512_setzero_ps());
            let lt = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(x.0, _mm512_set1_ps(1.0));
            Avx512(_mm512_maskz_mov_ps(gt & lt, v.0))
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub use arm::Neon;

#[cfg(target_arch = "aarch64")]
mod arm {
    use super::Simd;
    use std::arch::aarch64::*;

    #[derive(Clone, Copy)]
    pub struct Neon(float32x4_t);

    impl Simd for Neon {
        const W: usize = 4;
        #[inline(always)]
        unsafe fn zero() -> Self {
            Neon(vdupq_n_f32(0.0))
        }
        #[inline(always)]
        unsafe fn splat(x: f32) -> Self {
            Neon(vdupq_n_f32(x))
        }
        #[inline(always)]
        unsafe fn load(p: *const f32) -> Self {
            Neon(vld1q_f32(p))
        }
        #[inline(always)]
        unsafe fn store(self, p: *mut f32) {
            vst1q_f32(p, self.0)
        }
        #[inline(always)]
        unsafe fn add(self, o: Self) -> Self {
            Neon(vaddq_f32(self.0, o.0))
        }
        #[inline(always)]
        unsafe fn mul(self, o: Self) -> Self {
            Neon(vmulq_f32(self.0, o.0))
        }
        #[inline(always)]
        unsafe fn clamp01(self) -> Self {
            Neon(vminq_f32(vmaxq_f32(self.0, vdupq_n_f32(0.0)), vdupq_n_f32(1.0)))
        }
        #[inline(always)]
        unsafe fn select_in01(x: Self, v: Self) -> Self {
            let m = vandq_u32(vcgtq_f32(x.0, vdupq_n_f32(0.0)), vcltq_f32(x.0, vdupq_n_f32(1.0)));
            Neon(vreinterpretq_f32_u32(vandq_u32(m, vreinterpretq_u32_f32(v.0))))
        }
    }
}
