//! Zeroed heap buffers for big randomly-accessed tables (the TT; the NNUE
//! feature transformer measured no gain at 12.6 MB, where the L2 TLB still
//! covers its hot rows). Buffers of 2 MB or more start on a 2 MB boundary
//! and, on Linux, are `madvise(MADV_HUGEPAGE)`'d *before* they are first
//! written, so transparent huge pages back them when the system allows it
//! (`/sys/kernel/mm/transparent_hugepage/enabled` = `[madvise]`, the common
//! default, or `[always]`). With 4 KB pages nearly every TT probe misses the
//! TLB. Pure performance: contents and lookups are unchanged, and a refused
//! hint (THP off, no free huge pages, other OS) just leaves 4 KB pages.

use std::alloc::{alloc, dealloc, handle_alloc_error, Layout};
use std::ptr::NonNull;

const HUGE: usize = 2 << 20;

pub(crate) struct HugeBuf<T> {
    ptr: NonNull<T>,
    len: usize,
    layout: Layout,
}

// SAFETY: HugeBuf owns its elements like a Box<[T]>.
unsafe impl<T: Send> Send for HugeBuf<T> {}
// SAFETY: as above; shared access only hands out &[T].
unsafe impl<T: Sync> Sync for HugeBuf<T> {}

impl<T> HugeBuf<T> {
    /// `len` zeroed elements.
    ///
    /// # Safety
    /// The all-zero bit pattern must be a valid `T`.
    pub(crate) unsafe fn zeroed(len: usize) -> Self {
        let bytes = (len * std::mem::size_of::<T>()).max(1);
        let align = if bytes >= HUGE { HUGE } else { std::mem::align_of::<T>().max(64) };
        let layout = Layout::from_size_align(bytes, align).expect("buffer too large");
        // SAFETY: `layout` has nonzero size.
        let p = unsafe { alloc(layout) };
        if p.is_null() {
            handle_alloc_error(layout);
        }
        // Not alloc_zeroed: for this alignment it zero-fills with memset,
        // which would fault the pages in (as 4 KB pages) before the advice.
        if bytes >= HUGE {
            advise_huge(p, bytes);
        }
        // SAFETY: `p` is valid for `bytes` writes.
        unsafe { std::ptr::write_bytes(p, 0, bytes) };
        Self { ptr: NonNull::new(p as *mut T).unwrap(), len, layout }
    }
}

impl<T> Drop for HugeBuf<T> {
    fn drop(&mut self) {
        // SAFETY: allocated in `zeroed` with this layout; elements are dropped
        // first (no-op for the plain-data types used here).
        unsafe {
            std::ptr::drop_in_place(std::ptr::slice_from_raw_parts_mut(self.ptr.as_ptr(), self.len));
            dealloc(self.ptr.as_ptr() as *mut u8, self.layout);
        }
    }
}

impl<T> std::ops::Deref for HugeBuf<T> {
    type Target = [T];
    #[inline(always)]
    fn deref(&self) -> &[T] {
        // SAFETY: `len` initialised elements live at `ptr`.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<T> std::ops::DerefMut for HugeBuf<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: as in `deref`, and `&mut self` is exclusive.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

#[cfg(target_os = "linux")]
fn advise_huge(ptr: *mut u8, bytes: usize) {
    extern "C" {
        fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32;
    }
    const MADV_HUGEPAGE: i32 = 14;
    // SAFETY: [ptr, ptr + bytes) is a live allocation starting on a 2 MB
    // boundary (the advised range is rounded down into it); madvise only
    // changes paging policy and failure is harmless.
    unsafe {
        madvise(ptr as *mut std::ffi::c_void, bytes / HUGE * HUGE, MADV_HUGEPAGE);
    }
}

#[cfg(not(target_os = "linux"))]
fn advise_huge(_ptr: *mut u8, _bytes: usize) {}
