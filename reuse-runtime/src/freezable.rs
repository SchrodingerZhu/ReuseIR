#[repr(C)]
pub struct RegionCtx {
    pub(crate) tail: *mut FreezableRcBoxHeader,
}

#[repr(C)]
union FreezingStatus {
    ptr: *mut FreezableRcBoxHeader,
    scalar: usize,
}

enum StatusKind {
    Unmarked,
    Representative,
    Rank,
    ReferenceCount,
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
                1 => StatusKind::ReferenceCount,
                _ => StatusKind::Disposing,
            }
        }
    }
    pub fn get_value(self) -> usize {
        unsafe { (self.scalar & (isize::MAX as usize)) >> 2 as usize }
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
    pub fn reference_count(value: usize) -> Self {
        let mask = isize::MIN as usize;
        Self {
            scalar: mask | (value << 2) | 0b01,
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
pub struct FreezableRcBoxHeader {
    count: FreezingStatus,
    next: *mut Self,
    drop: std::option::Option<fn(*mut Self)>,
}
