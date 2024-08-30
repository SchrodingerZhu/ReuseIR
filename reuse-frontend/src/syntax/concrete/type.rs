use std::collections::HashMap;

use crate::syntax::concrete::{Decl, Def, Expr, File};
use crate::syntax::r#abstract::convert::convert;
use crate::syntax::r#abstract::{
    Decl as WellTypedDecl, Def as WellTypedDef, File as WellTypedFile, Term,
};
use crate::syntax::{Ctor, CtorParams, DataDef, FnDef, FnSig, Ident, Param, ID};

#[allow(dead_code)]
#[derive(Debug)]
enum Error<'src> {
    MismatchedType { want: Term<'src>, got: Term<'src> },
    MismatchedEffect { want: Term<'src>, got: Term<'src> },
    ExpectedFnType { typ: Term<'src> },
}

#[allow(dead_code)]
struct Inferred<'src> {
    term: Term<'src>,
    eff: Term<'src>,
    typ: Term<'src>,
}

impl<'src> Inferred<'src> {
    fn pure(term: Term<'src>, typ: Term<'src>) -> Self {
        let eff = Term::Pure;
        Self { term, eff, typ }
    }

    fn r#type(typ: Term<'src>) -> Self {
        Self::pure(typ, Term::Type)
    }
}

#[allow(dead_code)]
#[derive(Default)]
struct Checker<'src> {
    globals: HashMap<ID, WellTypedDef<'src>>,
    locals: HashMap<ID, Term<'src>>,
}

#[allow(dead_code)]
impl<'src> Checker<'src> {
    pub fn file(mut self, f: File<'src>) -> Result<WellTypedFile<'src>, Error<'src>> {
        Ok(WellTypedFile {
            decls: f
                .decls
                .into_vec()
                .into_iter()
                .map(|d| self.decl(d))
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .map(|name| WellTypedDecl {
                    name,
                    def: self.globals.remove(&name.id).unwrap(),
                })
                .collect(),
        })
    }

    fn decl(&mut self, decl: Decl<'src>) -> Result<Ident<'src>, Error<'src>> {
        self.locals.clear();
        let id = decl.name.id;
        match decl.def {
            Def::Fn(d) => self.fn_def(id, d)?,
            Def::Data(d) => self.data_def(id, d)?,
        }
        Ok(decl.name)
    }

    fn fn_def(&mut self, id: ID, def: FnDef<'src, Expr<'src>>) -> Result<(), Error<'src>> {
        let FnDef {
            sig:
                FnSig {
                    typ_params,
                    val_params,
                    eff,
                    ret,
                },
            body,
        } = def;
        let typ_params = self.params(typ_params)?;
        let val_params = self.params(val_params)?;
        let eff = self.check_type(&eff)?;
        let ret = self.check_type(&ret)?;
        let expected_eff = *eff.clone();
        let expected_typ = *ret.clone();
        self.globals.insert(
            id,
            WellTypedDef::UndefFn(FnSig {
                typ_params,
                val_params,
                eff,
                ret,
            }),
        );

        let body = self.check(&body, expected_eff, expected_typ)?;
        let sig = match self.globals.remove(&id).unwrap() {
            WellTypedDef::UndefFn(sig) => sig,
            _ => unreachable!(),
        };
        self.globals
            .insert(id, WellTypedDef::Fn(FnDef { sig, body }));

        Ok(())
    }

    fn data_def(&mut self, id: ID, def: DataDef<'src, Expr<'src>>) -> Result<(), Error<'src>> {
        let DataDef { typ_params, ctors } = def;
        let typ_params = self.params(typ_params)?;
        let ctors = ctors
            .into_vec()
            .into_iter()
            .map(|c| self.ctor(c))
            .collect::<Result<_, _>>()?;
        self.globals
            .insert(id, WellTypedDef::Data(DataDef { typ_params, ctors }));
        Ok(())
    }

    fn ctor(
        &mut self,
        Ctor { name, params }: Ctor<'src, Expr<'src>>,
    ) -> Result<Ctor<'src, Term<'src>>, Error<'src>> {
        let params = match params {
            CtorParams::None => CtorParams::None,
            CtorParams::Unnamed(ts) => CtorParams::Unnamed(
                ts.into_vec()
                    .into_iter()
                    .map(|t| self.check_type(&t))
                    .collect::<Result<_, _>>()?,
            ),
            CtorParams::Named(ps) => CtorParams::Named(self.params(ps)?),
        };
        Ok(Ctor { name, params })
    }

    fn params(
        &mut self,
        ps: Box<[Param<'src, Expr<'src>>]>,
    ) -> Result<Box<[Param<'src, Term<'src>>]>, Error<'src>> {
        ps.into_vec().into_iter().map(|p| self.param(p)).collect()
    }

    fn param(
        &mut self,
        Param { name, typ }: Param<'src, Expr<'src>>,
    ) -> Result<Param<'src, Term<'src>>, Error<'src>> {
        let typ = self.check_type(&typ)?;
        self.locals.insert(name.id, *typ.clone());
        Ok(Param { name, typ })
    }

    fn check_type(&mut self, typ: &Expr<'src>) -> Result<Box<Term<'src>>, Error<'src>> {
        self.check(typ, Term::Pure, Term::Type)
    }

    fn check<'a: 'src>(
        &mut self,
        e: &Expr<'src>,
        eff: Term<'src>,
        typ: Term<'src>,
    ) -> Result<Box<Term<'src>>, Error<'src>> {
        match e {
            Expr::Fn { .. } => match typ {
                Term::FnType { .. } => todo!(),
                typ => return Err(Error::ExpectedFnType { typ }),
            },
            _ => {}
        }

        let Inferred {
            term,
            eff: got_eff,
            typ: got_typ,
        } = self.infer(e)?;
        convert(&eff, &got_eff)
            .then_some(())
            .ok_or(Error::MismatchedEffect {
                want: eff,
                got: got_eff,
            })?;
        convert(&typ, &got_typ)
            .then_some(())
            .ok_or(Error::MismatchedType {
                want: typ,
                got: got_typ,
            })?;

        Ok(Box::new(term))
    }

    fn infer(&mut self, e: &Expr<'src>) -> Result<Inferred<'src>, Error<'src>> {
        Ok(match e {
            Expr::Ident(_) => todo!(),
            Expr::Type => Inferred::r#type(Term::Type),
            Expr::NoneType => Inferred::r#type(Term::NoneType),
            Expr::None => Inferred::pure(Term::None, Term::NoneType),
            Expr::Boolean => Inferred::r#type(Term::Boolean),
            Expr::False => Inferred::pure(Term::False, Term::Boolean),
            Expr::True => Inferred::pure(Term::True, Term::Boolean),
            Expr::String => Inferred::r#type(Term::String),
            Expr::Str(s) => Inferred::pure(Term::Str(s), Term::String),
            Expr::F32 => Inferred::r#type(Term::F32),
            Expr::F64 => Inferred::r#type(Term::F64),
            Expr::Float(v) => Inferred::pure(Term::Float(*v), Term::F64),
            Expr::FnType { .. } => todo!(),
            Expr::Fn { .. } => unreachable!(),
            Expr::Pure => Inferred::r#type(Term::Pure),
        })
    }
}
