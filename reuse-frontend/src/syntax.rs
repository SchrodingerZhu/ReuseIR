use crate::name::Ident;

pub trait Syntax {}

#[allow(dead_code)]
pub struct Param<'src, T: Syntax> {
    pub name: Ident<'src>,
    pub typ: Box<T>,
}

#[allow(dead_code)]
pub struct Decl<'src, T: Syntax> {
    pub name: Ident<'src>,
    pub def: Def<'src, T>,
}

#[allow(dead_code)]
pub enum Def<'src, T: Syntax> {
    Fn(FnDef<'src, T>),
    Data(DataDef<'src, T>),
}

#[allow(dead_code)]
pub struct FnDef<'src, T: Syntax> {
    pub typ_params: Box<[Param<'src, T>]>,
    pub val_params: Box<[Param<'src, T>]>,
    pub eff: Box<T>,
    pub ret: Box<T>,
    pub body: Box<T>,
}

#[allow(dead_code)]
pub struct DataDef<'src, T: Syntax> {
    pub typ_params: Box<[Param<'src, T>]>,
    pub ctors: Box<[Ctor<'src, T>]>,
}

#[allow(dead_code)]
pub struct Ctor<'src, T: Syntax> {
    pub name: Ident<'src>,
    pub params: CtorParams<'src, T>,
}

#[allow(dead_code)]
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
