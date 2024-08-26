use std::fmt::{Display, Formatter};

pub type ID = u64;

#[derive(Debug, Clone)]
pub struct Ident<'src> {
    pub id: ID,
    pub raw: &'src str,
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
