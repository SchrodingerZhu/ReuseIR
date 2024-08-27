use crate::syntax::{Ctor as C, CtorParams as CP, Decl as D, File as F, Ident, Param as P, Syntax};

#[allow(dead_code)]
pub type File<'src> = F<'src, Term<'src>>;
#[allow(dead_code)]
pub type Decl<'src> = D<'src, Term<'src>>;
#[allow(dead_code)]
pub type Param<'src> = P<'src, Term<'src>>;
#[allow(dead_code)]
pub type Ctor<'src> = C<'src, Term<'src>>;
#[allow(dead_code)]
pub type CtorParams<'src> = CP<'src, Term<'src>>;

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
        param: Param<'src>,
        body: Box<Term<'src>>,
    },
    GenericFn {
        param: Param<'src>,
        body: Box<Term<'src>>,
    },

    Pure,
}

impl<'src> Syntax for Term<'src> {}
