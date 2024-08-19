use crate::name::Ident;
use crate::syntax::{Decl, Param, Syntax};

#[allow(dead_code)]
pub type ExprParam<'src> = Param<'src, Expr<'src>>;

#[allow(dead_code)]
pub struct File<'src> {
    decls: Box<[Decl<'src, Expr<'src>>]>,
}

#[allow(dead_code)]
pub enum Expr<'src> {
    Ident(Ident<'src>),

    Type,

    Unit,
    UnitValue,

    Boolean,
    Bool(bool),

    String,
    Str(&'src str),

    F32,
    F64,
    Float(f64),

    Pure,
}

impl<'src> Syntax for Expr<'src> {}
