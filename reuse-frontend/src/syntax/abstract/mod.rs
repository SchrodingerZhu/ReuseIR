use crate::syntax::{DataDef, FnDef, FnSig, Ident, Param, Syntax};

pub mod convert;

#[allow(dead_code)]
#[derive(Debug)]
pub struct File<'src> {
    pub decls: Box<[Decl<'src>]>,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct Decl<'src> {
    pub name: Ident<'src>,
    pub def: Def<'src>,
}

#[allow(dead_code)]
#[derive(Debug)]
pub enum Def<'src> {
    Fn(FnDef<'src, Term<'src>>),
    UndefFn(FnSig<'src, Term<'src>>),
    Data(DataDef<'src, Term<'src>>),
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum Term<'src> {
    Ident(Ident<'src>),

    Type,

    NoneType,
    None,

    Boolean,
    False,
    True,

    String,
    Str(&'src str),

    F32,
    F64,
    Float(f64),

    FnType {
        param_types: Box<[Box<Term<'src>>]>,
        eff: Box<Term<'src>>,
        ret: Box<Term<'src>>,
    },
    Fn {
        params: Box<[Ident<'src>]>,
        body: Box<Term<'src>>,
    },
    GenericFnType {
        param: Param<'src, Term<'src>>,
        body: Box<Term<'src>>,
    },
    GenericFn {
        param: Param<'src, Term<'src>>,
        body: Box<Term<'src>>,
    },

    Pure,
}

impl<'src> Syntax for Term<'src> {}
