use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

pub mod r#abstract;
pub mod concrete;
pub mod surface;

pub type ID = u64;

pub type NameMap<'src> = HashMap<&'src str, ID>;

#[derive(Debug, Copy, Clone)]
pub struct Ident<'src> {
    pub raw: &'src str,
    pub id: ID,
}

pub fn fresh() -> ID {
    static NEXT_ID: AtomicU64 = AtomicU64::new(1);
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

#[allow(dead_code)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum PrimitiveType {
    Type,
    None,
    Bool,
    Str,
    F32,
    F64,
    Pure,
}

impl<'src> From<Primitive<'src>> for PrimitiveType {
    fn from(v: Primitive<'src>) -> Self {
        match v {
            Primitive::None => Self::None,
            Primitive::False | Primitive::True => Self::Bool,
            Primitive::Str(_) => Self::Str,
            Primitive::Float(_) => Self::F64,
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Copy, Clone)]
pub enum Primitive<'src> {
    None,
    False,
    True,
    Str(&'src str),
    Float(f64),
}

pub trait Syntax<'src>: Sized + From<Primitive<'src>> + From<PrimitiveType> {}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Param<'src, T: Syntax<'src>> {
    pub name: Ident<'src>,
    pub typ: Box<T>,
}

impl<'src, T: Syntax<'src>> Param<'src, T> {
    pub fn type_param(name: Ident<'src>) -> Self {
        Self {
            name,
            typ: Box::new(T::from(PrimitiveType::Type)),
        }
    }
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct FnSig<'src, T: Syntax<'src>> {
    pub typ_params: Box<[Param<'src, T>]>,
    pub val_params: Box<[Param<'src, T>]>,
    pub eff: Box<T>,
    pub ret: Box<T>,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct FnDef<'src, T: Syntax<'src>> {
    pub sig: FnSig<'src, T>,
    pub body: Box<T>,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct DataDef<'src, T: Syntax<'src>> {
    pub typ_params: Box<[Param<'src, T>]>,
    pub ctors: Box<[Ctor<'src, T>]>,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct Ctor<'src, T: Syntax<'src>> {
    pub name: Ident<'src>,
    pub params: CtorParams<'src, T>,
}

#[allow(dead_code)]
#[derive(Debug)]
pub enum CtorParams<'src, T: Syntax<'src>> {
    None,
    Unnamed(Box<[Box<T>]>),
    Named(Box<[Param<'src, T>]>),
}

impl<'src, T: Syntax<'src>> Default for CtorParams<'src, T> {
    fn default() -> Self {
        Self::None
    }
}
