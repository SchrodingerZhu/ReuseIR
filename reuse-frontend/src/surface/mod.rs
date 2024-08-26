use chumsky::error::Rich;
use chumsky::extra::Err;
use chumsky::prelude::{choice, just, none_of};
use chumsky::text::{digits, ident, inline_whitespace, newline, whitespace};
use chumsky::{IterParser, Parser};

use crate::concrete::{CtorExpr, CtorParamsExpr, Expr, File, ParamExpr};
use crate::name::{IDs, Ident};
use crate::syntax::{DataDef, Decl, Def, FnDef, Param};

#[cfg(test)]
mod tests;

#[allow(dead_code)]
#[derive(Default)]
struct Surface {
    ids: IDs,
}

pub type ErrMsg<'src> = Rich<'src, char>;

macro_rules! out {
    ($o:ty) => { impl Parser<'src, &'src str, $o, Err<ErrMsg<'src>>> };
}

macro_rules! primitive {
    ($kw:literal, $e:ident) => {
        just($kw).map(|_| Box::new(Expr::$e))
    };
}

#[allow(dead_code)]
impl Surface {
    fn file<'src>(&mut self) -> out!(File<'src>) {
        self.decl()
            .padded()
            .repeated()
            .collect::<Vec<_>>()
            .map(|decls| File {
                decls: decls.into_boxed_slice(),
            })
    }

    fn decl<'src>(&mut self) -> out!(Decl<'src, Expr<'src>>) {
        self.fn_decl().or(self.data_decl())
    }

    fn fn_decl<'src>(&mut self) -> out!(Decl<'src, Expr<'src>>) {
        just("def")
            .padded()
            .ignore_then(self.ident())
            .padded()
            .then(self.typ_params().or_not())
            .padded()
            .then(self.val_params())
            .padded()
            .then(
                just("->")
                    .padded()
                    .ignore_then(Self::type_expr())
                    .padded()
                    .or_not(),
            )
            .then_ignore(just(':'))
            .padded()
            .then(Self::expr())
            .padded_by(inline_whitespace())
            .then_ignore(newline())
            .map(|((((name, typ_params), val_params), ret), body)| Decl {
                name,
                def: Def::Fn(FnDef {
                    typ_params: typ_params.unwrap_or_default(),
                    val_params,
                    eff: Box::new(Expr::Pure), // TODO: parse effects
                    ret: ret.unwrap_or_else(|| Box::new(Expr::NoneType)),
                    body,
                }),
            })
    }

    fn data_decl<'src>(&mut self) -> out!(Decl<'src, Expr<'src>>) {
        just("data")
            .padded()
            .ignore_then(self.ident())
            .padded()
            .then(self.typ_params().or_not())
            .padded()
            .then_ignore(just(':'))
            .then_ignore(inline_whitespace())
            .then_ignore(newline())
            .then(self.ctors())
            .map(|((name, typ_params), ctors)| Decl {
                name,
                def: Def::Data(DataDef {
                    typ_params: typ_params.unwrap_or_default(),
                    ctors,
                }),
            })
    }

    fn ctors<'src>(&mut self) -> out!(Box<[CtorExpr<'src>]>) {
        self.ctor()
            .padded_by(inline_whitespace())
            .repeated()
            .at_least(1)
            .collect::<Vec<_>>()
            .map(Vec::into_boxed_slice)
    }

    fn ctor<'src>(&mut self) -> out!(CtorExpr<'src>) {
        self.ident()
            .then_ignore(inline_whitespace())
            .then(
                Self::ctor_unnamed_params()
                    .or(self.ctor_named_params())
                    .or_not()
                    .map(Option::unwrap_or_default),
            )
            .then_ignore(inline_whitespace())
            .then_ignore(newline())
            .map(|(name, params)| CtorExpr { name, params })
    }

    fn ctor_unnamed_params<'src>() -> out!(CtorParamsExpr<'src>) {
        just('(')
            .ignore_then(whitespace())
            .ignore_then(
                Self::type_expr()
                    .padded()
                    .separated_by(just(','))
                    .allow_trailing()
                    .at_least(1)
                    .collect::<Vec<_>>()
                    .map(Vec::into_boxed_slice)
                    .map(CtorParamsExpr::Unnamed),
            )
            .then_ignore(whitespace())
            .then_ignore(just(')'))
    }

    fn ctor_named_params<'src>(&mut self) -> out!(CtorParamsExpr<'src>) {
        just('(')
            .ignore_then(whitespace())
            .ignore_then(
                self.param()
                    .padded()
                    .separated_by(just(','))
                    .allow_trailing()
                    .at_least(1)
                    .collect::<Vec<_>>()
                    .map(Vec::into_boxed_slice)
                    .map(CtorParamsExpr::Named),
            )
            .then_ignore(whitespace())
            .then_ignore(just(')'))
    }

    fn expr<'src>() -> out!(Box<Expr<'src>>) {
        primitive!("None", None)
            .or(primitive!("False", True))
            .or(primitive!("True", True))
            .or(Self::str().map(Expr::Str).map(Box::new))
    }

    fn type_expr<'src>() -> out!(Box<Expr<'src>>) {
        primitive!("bool", Boolean)
            .or(primitive!("str", String))
            .or(primitive!("None", NoneType))
            .or(primitive!("f32", F32))
            .or(primitive!("f64", F64))
    }

    fn param<'src>(&mut self) -> out!(ParamExpr<'src>) {
        self.ident()
            .then_ignore(whitespace())
            .then_ignore(just(':'))
            .then_ignore(whitespace())
            .then(Self::type_expr())
            .map(|(name, typ)| Param { name, typ })
    }

    fn val_params<'src>(&mut self) -> out!(Box<[ParamExpr<'src>]>) {
        just('(')
            .ignore_then(whitespace())
            .ignore_then(
                self.param()
                    .padded()
                    .separated_by(just(','))
                    .allow_trailing()
                    .collect::<Vec<_>>()
                    .map(Vec::into_boxed_slice)
                    .or_not()
                    .map(Option::unwrap_or_default),
            )
            .then_ignore(whitespace())
            .then_ignore(just(')'))
    }

    fn typ_params<'src>(&mut self) -> out!(Box<[ParamExpr<'src>]>) {
        just('[')
            .ignore_then(whitespace())
            .ignore_then(
                self.ident()
                    .padded()
                    .map(|name| Param {
                        name,
                        typ: Box::new(Expr::Type),
                    })
                    .separated_by(just(','))
                    .allow_trailing()
                    .at_least(1)
                    .collect::<Vec<_>>()
                    .map(Vec::into_boxed_slice),
            )
            .then_ignore(whitespace())
            .then_ignore(just(']'))
    }

    fn ident<'src>(&mut self) -> out!(Ident<'src>) {
        let id = self.ids.fresh();
        ident().map(move |s| Ident::new(id, s))
    }

    fn str<'src>() -> out!(&'src str) {
        none_of("\\\"")
            .ignored()
            .or(Self::escaped())
            .repeated()
            .to_slice()
            .delimited_by(just('"'), just('"'))
    }

    fn escaped<'src>() -> out!(()) {
        const INVALID_CHAR: char = '\u{FFFD}';
        just('\\')
            .then(choice((
                just('\\'),
                just('/'),
                just('"'),
                just('b').to('\x08'),
                just('f').to('\x0C'),
                just('n').to('\n'),
                just('r').to('\r'),
                just('t').to('\t'),
                just('u').ignore_then(digits(16).exactly(4).to_slice().validate(
                    |digits, e, emitter| {
                        char::from_u32(u32::from_str_radix(digits, 16).unwrap()).unwrap_or_else(
                            || {
                                emitter.emit(Rich::custom(e.span(), "invalid unicode character"));
                                INVALID_CHAR
                            },
                        )
                    },
                )),
            )))
            .ignored()
    }
}
