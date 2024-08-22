#![allow(clippy::missing_safety_doc)]

use std::ptr::NonNull;

use smallvec::SmallVec;

const STATUS_COUNTER_PADDING_BITS: usize = 2;
#[repr(C)]
pub struct RegionCtx {
    pub(crate) tail: *mut FreezableRcBoxHeader,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union FreezingStatus {
    ptr: *mut FreezableRcBoxHeader,
    scalar: usize,
}

#[derive(PartialEq, Eq)]
pub enum StatusKind {
    Unmarked,
    Representative,
    Rank,
    Rc,
    Disposing,
}

impl FreezingStatus {
    pub fn get_kind(self) -> StatusKind {
        unsafe {
            if self.scalar == 0 {
                return StatusKind::Unmarked;
            }
            if self.scalar as isize > 0 {
                return StatusKind::Representative;
            }
            match self.scalar & 0b11 {
                0 => StatusKind::Rank,
                1 => StatusKind::Rc,
                _ => StatusKind::Disposing,
            }
        }
    }
    pub fn get_value(self) -> usize {
        unsafe { (self.scalar & (isize::MAX as usize)) >> 2 }
    }
    pub unsafe fn get_pointer_unchecked(self) -> NonNull<FreezableRcBoxHeader> {
        unsafe { NonNull::new_unchecked(self.ptr) }
    }
    pub fn unmarked() -> Self {
        Self { scalar: 0 }
    }
    pub fn representative(ptr: *mut FreezableRcBoxHeader) -> Self {
        Self { ptr }
    }
    pub fn rank(value: usize) -> Self {
        let mask = isize::MIN as usize;
        Self {
            scalar: mask | (value << STATUS_COUNTER_PADDING_BITS),
        }
    }
    pub fn rc(value: usize) -> Self {
        let mask = isize::MIN as usize;
        Self {
            scalar: mask | (value << STATUS_COUNTER_PADDING_BITS) | 0b01,
        }
    }
    pub fn disposing() -> Self {
        let mask = isize::MIN as usize;
        Self {
            scalar: mask | 0b10,
        }
    }
}

#[repr(C)]
pub struct FreezableVTable {
    drop: std::option::Option<unsafe extern "C" fn(*mut FreezableRcBoxHeader)>,
    size: usize,
    alignment: usize,
    scan_count: usize,
    scan_offset: [usize; 0],
}

#[repr(C)]
pub struct FreezableRcBoxHeader {
    status: FreezingStatus,
    next: *mut Self,
    vtable: *const FreezableVTable,
}

struct FieldIterator {
    slice_iter: std::slice::Iter<'static, usize>,
    base_pointer: NonNull<u8>,
}

impl FieldIterator {
    unsafe fn new(base_pointer: NonNull<FreezableRcBoxHeader>) -> Self {
        let vtable = base_pointer.as_ref().vtable;
        let slice =
            std::slice::from_raw_parts((*vtable).scan_offset.as_ptr(), (*vtable).scan_count);
        Self {
            slice_iter: slice.iter(),
            base_pointer: base_pointer.cast(),
        }
    }
}

impl Iterator for FieldIterator {
    type Item = Option<NonNull<FreezableRcBoxHeader>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.slice_iter.next().map(|offset| unsafe {
            let pointer = self
                .base_pointer
                .byte_add(*offset)
                .cast::<*mut FreezableRcBoxHeader>();
            NonNull::new(pointer.read())
        })
    }
}

unsafe fn increase_refcnt(mut object: NonNull<FreezableRcBoxHeader>, count: usize) {
    object.as_mut().status.scalar += count << STATUS_COUNTER_PADDING_BITS;
}

unsafe fn decrease_refcnt(mut object: NonNull<FreezableRcBoxHeader>, count: usize) -> bool {
    object.as_mut().status.scalar -= count << STATUS_COUNTER_PADDING_BITS;
    object.as_mut().status.scalar <= (isize::MIN as usize | 0b11)
}

unsafe fn find_representative(
    mut object: NonNull<FreezableRcBoxHeader>,
) -> NonNull<FreezableRcBoxHeader> {
    let mut root = object;
    while root.as_ref().status.get_kind() == StatusKind::Representative {
        root = root.as_ref().status.get_pointer_unchecked();
    }
    while object.as_ref().status.get_kind() == StatusKind::Representative {
        let next = object.as_ref().status.get_pointer_unchecked();
        object.as_mut().status = FreezingStatus::representative(root.as_ptr());
        object = next;
    }
    root
}

unsafe fn union(x: NonNull<FreezableRcBoxHeader>, y: NonNull<FreezableRcBoxHeader>) -> bool {
    let mut rep_x = find_representative(x);
    let mut rep_y = find_representative(y);
    if rep_x == rep_y {
        return false;
    }
    if rep_x.as_ref().status.get_value() < rep_y.as_ref().status.get_value() {
        std::mem::swap(&mut rep_x, &mut rep_y);
    }
    if rep_x.as_ref().status.get_value() == rep_y.as_ref().status.get_value() {
        rep_x.as_mut().status = FreezingStatus::rank(rep_x.as_ref().status.get_value() + 1);
    }
    rep_y.as_mut().status = FreezingStatus::representative(rep_x.as_ptr());
    true
}

#[cold]
unsafe fn dispose(object: NonNull<FreezableRcBoxHeader>) {
    type Stack = SmallVec<[NonNull<FreezableRcBoxHeader>; 16]>;
    unsafe fn add_stack(stack: &mut Stack, mut object: NonNull<FreezableRcBoxHeader>) {
        stack.push(object);
        object.as_mut().status = FreezingStatus::disposing();
    }
    let mut dfs = Stack::new();
    let mut scc = Stack::new();
    let mut recycle = Stack::new();
    add_stack(&mut dfs, find_representative(object));
    while let Some(obj) = dfs.pop() {
        scc.push(obj);
        while let Some(obj) = scc.pop() {
            recycle.push(obj);
            for field in FieldIterator::new(obj).flatten() {
                let next = find_representative(field);
                match next.as_ref().status.get_kind() {
                    StatusKind::Disposing if field != next => add_stack(&mut scc, field),
                    StatusKind::Rc if decrease_refcnt(next, 1) => add_stack(&mut dfs, next),
                    _ => continue,
                }
            }
        }
    }
    stacker::maybe_grow(16 * 1024, 1024 * 1024, || {
        while let Some(obj) = recycle.pop() {
            let vtable = obj.as_ref().vtable;
            if let Some(dtor) = (*vtable).drop {
                dtor(obj.as_ptr());
            }
            crate::allocator::__reuse_ir_dealloc(
                obj.cast().as_ptr(),
                (*vtable).size,
                (*vtable).alignment,
            );
        }
    });
}

#[no_mangle]
pub unsafe extern "C" fn __reuse_ir_freeze(object: *mut FreezableRcBoxHeader) {
    let Some(object) = NonNull::new(object) else {
        panic!("attempt to freeze null object");
    };
    type PendingList = SmallVec<[NonNull<FreezableRcBoxHeader>; 32]>;
    let mut pending = PendingList::new();
    unsafe fn freeze(mut object: NonNull<FreezableRcBoxHeader>, pending: &mut PendingList) {
        match object.as_ref().status.get_kind() {
            StatusKind::Rc => {
                let root = find_representative(object);
                increase_refcnt(root, 1);
            }
            StatusKind::Rank => loop {
                let Some(next) = pending.pop() else {
                    break;
                };
                if !union(object, next) {
                    break;
                }
            },
            StatusKind::Unmarked => {
                object.as_mut().status = FreezingStatus::rc(1);
                pending.push(object);
                stacker::maybe_grow(16 * 1024, 1024 * 1024, || {
                    let iter = FieldIterator::new(object);
                    for field in iter.flatten() {
                        freeze(field, pending);
                    }
                });
            }
            _ => std::hint::unreachable_unchecked(),
        }
    }
    freeze(object, &mut pending)
}

#[no_mangle]
pub unsafe extern "C" fn __reuse_ir_freeze_atomic(object: *mut FreezableRcBoxHeader) {
    unimplemented!("cannot freeze {object:?}")
}

#[no_mangle]
pub unsafe extern "C" fn __reuse_ir_acquire_freezable(
    object: *mut FreezableRcBoxHeader,
    count: usize,
) {
    let Some(object) = NonNull::new(object) else {
        panic!("attempt to acquire null object");
    };
    let root = find_representative(object);
    increase_refcnt(root, count);
}

#[no_mangle]
pub unsafe extern "C" fn __reuse_ir_release_freezable(
    object: *mut FreezableRcBoxHeader,
    count: usize,
) {
    if let Some(object) = NonNull::new(object) {
        if decrease_refcnt(object, count) {
            dispose(object);
        }
    } else {
        panic!("attempt to release null object");
    }
}

#[no_mangle]
pub unsafe extern "C" fn __reuse_ir_acquire_atomic_freezable(object: *mut FreezableRcBoxHeader) {
    unimplemented!("cannot acquire {object:?}")
}

#[no_mangle]
pub unsafe extern "C" fn __reuse_ir_release_atomic_freezable(object: *mut FreezableRcBoxHeader) {
    unimplemented!("cannot release {object:?}")
}

#[no_mangle]
pub unsafe extern "C" fn __reuse_ir_clean_up_region(region: *mut RegionCtx) {
    unimplemented!("cannot clean up region: {region:?}")
}

#[no_mangle]
pub unsafe extern "C" fn __reuse_ir_clean_up_region_atomic(region: *mut RegionCtx) {
    unimplemented!("cannot clean up region: {region:?}")
}
