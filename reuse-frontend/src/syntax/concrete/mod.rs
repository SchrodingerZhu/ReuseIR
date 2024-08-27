mod scope;

use crate::syntax::{Ctor, CtorParams, Decl, File, Ident, Param, Syntax};

#[allow(dead_code)]
pub type FileExpr<'src> = File<'src, Expr<'src>>;
#[allow(dead_code)]
pub type DeclExpr<'src> = Decl<'src, Expr<'src>>;
#[allow(dead_code)]
pub type ParamExpr<'src> = Param<'src, Expr<'src>>;
#[allow(dead_code)]
pub type CtorExpr<'src> = Ctor<'src, Expr<'src>>;
#[allow(dead_code)]
pub type CtorParamsExpr<'src> = CtorParams<'src, Expr<'src>>;

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum Expr<'src> {
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
        param_types: Box<[Box<Expr<'src>>]>,
        eff: Box<Expr<'src>>,
        ret: Box<Expr<'src>>,
    },
    Fn {
        params: Box<[Ident<'src>]>,
        body: Box<Expr<'src>>,
    },

    Pure,
}

impl<'src> Syntax for Expr<'src> {}