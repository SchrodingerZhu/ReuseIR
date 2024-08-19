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
    pub typ_params: Box<[Param<'src, T>]>,
    pub val_params: Box<[Param<'src, T>]>,
    pub eff: Box<T>,
    pub ret: Box<T>,
    pub def: Def<'src, T>,
}

#[allow(dead_code)]
pub enum Def<'src, T: Syntax> {
    Fn(FnDef<T>),
    Data(DataDef<'src, T>),
}

#[allow(dead_code)]
pub struct FnDef<T: Syntax> {
    pub body: Box<T>,
}

#[allow(dead_code)]
pub struct DataDef<'src, T: Syntax> {
    ctors: Box<[Ctor<'src, T>]>,
}

#[allow(dead_code)]
pub struct Ctor<'src, T: Syntax> {
    name: Ident<'src>,
    val_params: Box<[Param<'src, T>]>,
}
