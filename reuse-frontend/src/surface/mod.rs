use chumsky::error::Cheap;
use chumsky::extra::Err;
use chumsky::prelude::just;
use chumsky::text::ascii::keyword;
use chumsky::text::{ident, inline_whitespace, newline};
use chumsky::{IterParser, Parser};

use crate::concrete::{Expr, ParamExpr};
use crate::name::{IDs, Ident};
use crate::syntax::{Decl, Def, FnDef, Param};

#[cfg(test)]
mod tests;

#[allow(dead_code)]
#[derive(Default)]
struct Surface {
    ids: IDs,
}

macro_rules! out {
    ($o:ty) => { impl Parser<'src, &'src str, $o, Err<Cheap>> };
}

macro_rules! primitive {
    ($kw:literal, $e:ident) => {
        just($kw).map(|_| Box::new(Expr::$e))
    };
}

#[allow(dead_code)]
impl Surface {
    fn decl<'src>(&mut self) -> out!(Decl<'src, Expr<'src>>) {
        self.fn_decl()
    }

    fn fn_decl<'src>(&mut self) -> out!(Decl<'src, Expr<'src>>) {
        keyword("def")
            .padded()
            .ignore_then(self.ident())
            .padded()
            .then(
                self.ident()
                    .padded()
                    .map(|name| Param {
                        name,
                        typ: Box::new(Expr::Type),
                    })
                    .separated_by(just(','))
                    .at_least(1)
                    .collect::<Vec<_>>()
                    .delimited_by(just('['), just(']'))
                    .or_not(),
            )
            .padded()
            .then(
                self.param()
                    .padded()
                    .separated_by(just(','))
                    .collect::<Vec<_>>()
                    .delimited_by(just('('), just(')')),
            )
            .padded()
            .then(just("->").padded().ignore_then(self.type_expr()).or_not())
            .padded()
            .then_ignore(just(':'))
            .padded()
            .then(self.expr())
            .padded_by(inline_whitespace())
            .then_ignore(newline())
            .map(|((((name, typ_params), val_params), ret), body)| Decl {
                name,
                typ_params: typ_params.unwrap_or_default().into_boxed_slice(),
                val_params: val_params.into_boxed_slice(),
                eff: Box::new(Expr::Pure), // TODO: parse effects
                ret: ret.unwrap_or_else(|| Box::new(Expr::NoneType)),
                def: Def::Fn(FnDef { body }),
            })
    }

    fn expr<'src>(&mut self) -> out!(Box<Expr<'src>>) {
        primitive!("None", None)
            .or(primitive!("False", True))
            .or(primitive!("True", True))
    }

    fn type_expr<'src>(&mut self) -> out!(Box<Expr<'src>>) {
        primitive!("bool", Boolean)
            .or(primitive!("str", String))
            .or(primitive!("None", NoneType))
            .or(primitive!("f32", F32))
            .or(primitive!("f64", F64))
    }

    fn param<'src>(&mut self) -> out!(ParamExpr<'src>) {
        self.ident()
            .padded()
            .then_ignore(just(':'))
            .padded()
            .then(self.type_expr())
            .map(|(name, typ)| Param { name, typ })
    }

    fn ident<'src>(&mut self) -> out!(Ident<'src>) {
        let id = self.ids.fresh();
        ident().map(move |s| Ident::new(id, s))
    }
}
