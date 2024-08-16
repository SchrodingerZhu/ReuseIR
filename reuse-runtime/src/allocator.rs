use std::ffi::c_void;

#[no_mangle]
pub unsafe extern "C" fn __reuse_ir_alloc(size: usize, alignment: usize) -> *mut c_void {
    std::alloc::alloc(std::alloc::Layout::from_size_align_unchecked(
        size, alignment,
    )) as _
}

#[no_mangle]
pub unsafe extern "C" fn __reuse_ir_dealloc(ptr: *mut u8, size: usize, alignment: usize) {
    std::alloc::dealloc(
        ptr as _,
        std::alloc::Layout::from_size_align_unchecked(size, alignment),
    )
}
