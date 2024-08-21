use std::fmt::{Display, Formatter};

pub type ID = u64;

pub struct Ident<'src> {
    id: ID,
    raw: &'src str,
}

impl<'src> Ident<'src> {
    pub fn new(id: ID, raw: &'src str) -> Self {
        Self { id, raw }
    }

    pub fn unbound() -> Self {
        Self { id: 0, raw: "_" }
    }
}

impl<'src> Display for Ident<'src> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}@{}", self.raw, self.id)
    }
}

#[derive(Default)]
pub struct IDs(ID);

impl IDs {
    pub fn fresh(&mut self) -> ID {
        self.0 += 1;
        self.0
    }
}
