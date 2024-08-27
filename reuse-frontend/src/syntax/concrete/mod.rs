mod scope;
mod r#type;

use crate::syntax::{Ctor as C, CtorParams as CP, Decl as D, File as F, Ident, Param as P, Syntax};

#[allow(dead_code)]
pub type File<'src> = F<'src, Expr<'src>>;
#[allow(dead_code)]
pub type Decl<'src> = D<'src, Expr<'src>>;
#[allow(dead_code)]
pub type Param<'src> = P<'src, Expr<'src>>;
#[allow(dead_code)]
pub type Ctor<'src> = C<'src, Expr<'src>>;
#[allow(dead_code)]
pub type CtorParams<'src> = CP<'src, Expr<'src>>;

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
