use chumsky::error::Rich;
use chumsky::extra::Err;
use chumsky::prelude::{choice, just, none_of, one_of};
use chumsky::recursive::recursive;
use chumsky::text::{digits, ident, inline_whitespace, int, newline, whitespace};
use chumsky::{IterParser, Parser};

use crate::syntax::concrete::{CtorExpr, CtorParamsExpr, Expr, FileExpr, ParamExpr};
use crate::syntax::{DataDef, Decl, Def, FnDef, IDs, Ident, Param};

#[allow(dead_code)]
#[derive(Default)]
struct Surface {
    ids: IDs,
}

pub type Msg<'src> = Rich<'src, char>;

macro_rules! out {
    ($o:ty) => { impl Parser<'src, &'src str, $o, Err<Msg<'src>>> };
}

macro_rules! primitive {
    ($kw:literal, $e:ident) => {
        just($kw).map(|_| Expr::$e)
    };
}

#[allow(dead_code)]
impl Surface {
    fn file<'src>(&mut self) -> out!(FileExpr<'src>) {
        self.decl()
            .padded()
            .repeated()
            .collect::<Vec<_>>()
            .map(|decls| FileExpr {
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
            .then(self.expr())
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

    fn expr<'src>(&mut self) -> out!(Box<Expr<'src>>) {
        recursive(|expr| {
            let lambda = just("lambda")
                .ignore_then(whitespace())
                .ignore_then(
                    self.ident()
                        .padded()
                        .separated_by(just(','))
                        .collect::<Vec<_>>()
                        .map(Vec::into_boxed_slice),
                )
                .then_ignore(whitespace())
                .then_ignore(just(':'))
                .then_ignore(whitespace())
                .then(expr)
                .map(|(params, body)| Expr::Fn { params, body });

            lambda
                .or(primitive!("None", None))
                .or(primitive!("False", True))
                .or(primitive!("True", True))
                .or(Self::str().map(Expr::Str))
                .or(Self::float().map(Expr::Float))
                .map(Box::new)
                .boxed()
        })
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

    fn float<'src>() -> out!(f64) {
        just('-')
            .or_not()
            .then(int(10))
            .then(just('.').then(Self::decimal()).or_not())
            .then(Self::exponential().or_not())
            .to_slice()
            .map(|s: &str| s.parse().unwrap())
    }

    fn decimal<'src>() -> out!(&'src str) {
        digits(10).to_slice()
    }

    fn exponential<'src>() -> out!(&'src str) {
        just('e')
            .or(just('E'))
            .then(one_of("+-").or_not())
            .then(Self::decimal())
            .to_slice()
    }

    fn type_expr<'src>() -> out!(Box<Expr<'src>>) {
        recursive(|expr| {
            let fn_type = just('[')
                .ignore_then(whitespace())
                .ignore_then(
                    expr.clone()
                        .padded()
                        .separated_by(just(','))
                        .collect::<Vec<_>>()
                        .map(Vec::into_boxed_slice),
                )
                .then_ignore(whitespace())
                .then_ignore(just(']'))
                .then_ignore(whitespace())
                .then_ignore(just("->"))
                .then_ignore(whitespace())
                .then(expr)
                .map(|(param_types, ret)| {
                    Expr::FnType {
                        param_types,
                        eff: Box::new(Expr::Pure), // TODO: parse effects
                        ret,
                    }
                });

            fn_type
                .or(primitive!("bool", Boolean))
                .or(primitive!("str", String))
                .or(primitive!("None", NoneType))
                .or(primitive!("f32", F32))
                .or(primitive!("f64", F64))
                .map(Box::new)
                .boxed()
        })
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
        ident().map(move |raw| Ident { raw, id })
    }
}

#[cfg(test)]
mod tests {
    use chumsky::Parser;

    use crate::print_parse_errors;
    use crate::syntax::surface::Surface;

    #[test]
    fn it_parses_file() {
        const SRC: &str = "

def f0 (  ) -> None : None
def f1 (
) -> None : None

def f2 [
    T,
    U,
] (s: str) -> str :
\"hello, world\"

def f3() -> f64: -114.514

def f4() -> [] -> bool:
    lambda
        :
            True
def f5() -> [bool, str] -> bool: lambda a, b: False

data
  Foo [T]:
    A
B(str)
    C(
        s  : str  ,
        b: bool  ,
    )

";
        Surface::default()
            .file()
            .parse(SRC)
            .into_result()
            .inspect_err(|es| print_parse_errors(es, SRC))
            .unwrap();
    }
}