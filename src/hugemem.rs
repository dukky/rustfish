//! Zeroed buffers whose working range starts on a 2 MB boundary, with
//! transparent huge pages requested for that range on Linux (`madvise`), so
//! the TT and the NNUE feature transformer cost far fewer TLB misses. Other
//! targets get the same layout without the hint. Pure performance: the
//! contents and every lookup are unchanged.

use std::alloc::{alloc_zeroed, handle_alloc_error, Layout};

const HUGE: usize = 2 << 20;

/// A boxed slice of at least `n` zeroed `T`s and the index `base` at which
/// `n` elements start on a 2 MB boundary (`box[base..base + n]`).
///
/// # Safety
/// The all-zero bit pattern must be a valid `T`, and `size_of::<T>()` must be
/// a nonzero divisor of 2 MB not smaller than `align_of::<T>()`.
pub(crate) unsafe fn zeroed_aligned<T>(n: usize) -> (Box<[T]>, usize) {
    let size = std::mem::size_of::<T>();
    assert!(size > 0 && HUGE % size == 0 && size >= std::mem::align_of::<T>());
    let extra = HUGE / size;
    let len = n + extra;
    let layout = Layout::array::<T>(len).expect("buffer too large");
    let ptr = alloc_zeroed(layout) as *mut T;
    if ptr.is_null() {
        handle_alloc_error(layout);
    }
    // `ptr` is a multiple of size_of::<T>() (allocations of an array of T are
    // T-aligned and size is a power-of-two divisor of 2 MB), so the distance
    // to the next 2 MB boundary is a whole number of elements.
    let misalign = ptr as usize % HUGE;
    let base = if misalign == 0 { 0 } else { (HUGE - misalign) / size };
    advise_huge(ptr.add(base) as *mut u8, n * size);
    (Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len)), base)
}

#[cfg(target_os = "linux")]
fn advise_huge(ptr: *mut u8, bytes: usize) {
    extern "C" {
        fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32;
    }
    const MADV_HUGEPAGE: i32 = 14;
    let len = bytes / HUGE * HUGE;
    if len > 0 {
        // SAFETY: [ptr, ptr + len) lies inside a live allocation; madvise only
        // changes paging policy (failure, e.g. THP disabled, is harmless).
        unsafe {
            madvise(ptr as *mut std::ffi::c_void, len, MADV_HUGEPAGE);
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn advise_huge(_ptr: *mut u8, _bytes: usize) {}
