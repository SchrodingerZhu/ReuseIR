#![allow(clippy::missing_safety_doc)]
use std::ffi::c_void;

#[no_mangle]
pub unsafe extern "C" fn __reuse_ir_alloc(size: usize, alignment: usize) -> *mut c_void {
    std::alloc::alloc(std::alloc::Layout::from_size_align_unchecked(
        size, alignment,
    )) as _
}

#[no_mangle]
pub unsafe extern "C" fn __reuse_ir_dealloc(ptr: *mut c_void, size: usize, alignment: usize) {
    std::alloc::dealloc(
        ptr as _,
        std::alloc::Layout::from_size_align_unchecked(size, alignment),
    )
}

#[no_mangle]
pub unsafe extern "C" fn __reuse_ir_realloc(
    ptr: *mut c_void,
    old_size: usize,
    alignment: usize,
    size: usize,
) -> *mut c_void {
    std::alloc::realloc(
        ptr as _,
        std::alloc::Layout::from_size_align_unchecked(old_size, alignment),
        size,
    ) as _
}
