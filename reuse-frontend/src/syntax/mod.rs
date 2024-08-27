use std::fmt::{Display, Formatter};

pub mod concrete;
pub mod surface;

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

pub trait Syntax {}

#[allow(dead_code)]
#[derive(Debug)]
pub struct Param<'src, T: Syntax> {
    pub name: Ident<'src>,
    pub typ: Box<T>,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct File<'src, T: Syntax> {
    pub decls: Box<[Decl<'src, T>]>,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct Decl<'src, T: Syntax> {
    pub name: Ident<'src>,
    pub def: Def<'src, T>,
}

#[allow(dead_code)]
#[derive(Debug)]
pub enum Def<'src, T: Syntax> {
    Fn(FnDef<'src, T>),
    Data(DataDef<'src, T>),
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct FnDef<'src, T: Syntax> {
    pub typ_params: Box<[Param<'src, T>]>,
    pub val_params: Box<[Param<'src, T>]>,
    pub eff: Box<T>,
    pub ret: Box<T>,
    pub body: Box<T>,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct DataDef<'src, T: Syntax> {
    pub typ_params: Box<[Param<'src, T>]>,
    pub ctors: Box<[Ctor<'src, T>]>,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct Ctor<'src, T: Syntax> {
    pub name: Ident<'src>,
    pub params: CtorParams<'src, T>,
}

#[allow(dead_code)]
#[derive(Debug)]
pub enum CtorParams<'src, T: Syntax> {
    None,
    Unnamed(Box<[Box<T>]>),
    Named(Box<[Param<'src, T>]>),
}

impl<'src, T: Syntax> Default for CtorParams<'src, T> {
    fn default() -> Self {
        Self::None
    }
}
