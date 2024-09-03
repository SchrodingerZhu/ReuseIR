use crate::syntax::{DataDef, FnDef, Ident, Primitive, PrimitiveType, Syntax};

mod scope;
mod r#type;

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
    Fn(FnDef<'src, Expr<'src>>),
    Data(DataDef<'src, Expr<'src>>),
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum Expr<'src> {
    PrimitiveType(PrimitiveType),
    Primitive(Primitive<'src>),

    Ident(Ident<'src>),

    FnType {
        param_types: Box<[Box<Self>]>,
        eff: Box<Self>,
        ret: Box<Self>,
    },
    Fn {
        params: Box<[Ident<'src>]>,
        body: Box<Self>,
    },
    Call {
        f: Box<Self>,
        typ_args: Box<[Box<Self>]>,
        val_args: Box<[Box<Self>]>,
    },
}

impl<'src> From<Primitive<'src>> for Expr<'src> {
    fn from(v: Primitive<'src>) -> Self {
        Self::Primitive(v)
    }
}

impl<'src> From<PrimitiveType> for Expr<'src> {
    fn from(t: PrimitiveType) -> Self {
        Self::PrimitiveType(t)
    }
}

impl<'src> Syntax<'src> for Expr<'src> {}
