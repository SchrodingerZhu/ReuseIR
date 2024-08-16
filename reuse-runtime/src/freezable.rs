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

pub enum StatusKind {
    Unmarked,
    Representative,
    Rank,
    Rc,
    FrozenRc,
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
                2 => StatusKind::FrozenRc,
                _ => StatusKind::Disposing,
            }
        }
    }
    pub fn get_value(self) -> usize {
        unsafe { (self.scalar & (isize::MAX as usize)) >> 2 }
    }
    pub fn get_pointer(self) -> *mut FreezableRcBoxHeader {
        unsafe { self.ptr }
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
            scalar: mask | (value << 2) | 0b00,
        }
    }
    pub fn rc(value: usize) -> Self {
        let mask = isize::MIN as usize;
        Self {
            scalar: mask | (value << 2) | 0b01,
        }
    }
    pub fn frozen_rc(value: usize) -> Self {
        let mask = isize::MIN as usize;
        Self {
            scalar: mask | (value << 2) | 0b010,
        }
    }
    pub fn disposing() -> Self {
        let mask = isize::MIN as usize;
        Self {
            scalar: mask | 0b11,
        }
    }
}

#[repr(C)]
pub struct FreezableVTable {
    drop: std::option::Option<unsafe extern "C" fn(*mut FreezableRcBoxHeader)>,
    scan_count: usize,
    scan_offset: [usize; 0],
}

#[repr(C)]
pub struct FreezableRcBoxHeader {
    status: FreezingStatus,
    next: *mut Self,
    vtable: *const FreezableVTable,
}

unsafe fn find_representative(object: *mut FreezableRcBoxHeader) -> *mut FreezableRcBoxHeader {
    let status = (*object).status;
    match status.get_kind() {
        StatusKind::Representative => {
            let parent = find_representative(status.get_pointer());
            (*object).status = FreezingStatus::representative(parent);
            parent
        }
        _ => object,
    }
}

unsafe fn union(x: *mut FreezableRcBoxHeader, y: *mut FreezableRcBoxHeader) {
    let mut rep_x = find_representative(x);
    let mut rep_y = find_representative(y);
    if rep_x == rep_y {
        return;
    }
    if (*rep_x).status.get_value() < (*rep_y).status.get_value() {
        std::mem::swap(&mut rep_x, &mut rep_y);
    }
    if (*rep_x).status.get_value() == (*rep_y).status.get_value() {
        (*rep_x).status = FreezingStatus::rank((*rep_x).status.get_value() + 1);
    }
    (*rep_y).status = FreezingStatus::representative(rep_x);
}

#[no_mangle]
pub unsafe extern "C" fn __reuse_ir_freeze(object: *mut FreezableRcBoxHeader) {
    unimplemented!("cannot freeze {object:?}")
}

#[no_mangle]
pub unsafe extern "C" fn __reuse_ir_freeze_atomic(object: *mut FreezableRcBoxHeader) {
    unimplemented!("cannot freeze {object:?}")
}

#[no_mangle]
pub unsafe extern "C" fn __reuse_ir_acquire_freezable(object: *mut FreezableRcBoxHeader) {
    unimplemented!("cannot acquire {object:?}")
}

#[no_mangle]
pub unsafe extern "C" fn __reuse_ir_release_freezable(object: *mut FreezableRcBoxHeader) {
    unimplemented!("cannot release {object:?}")
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
