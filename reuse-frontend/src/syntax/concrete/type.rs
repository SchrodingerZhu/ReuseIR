use std::collections::HashMap;

use crate::syntax::concrete::{Decl, Def, Expr, File};
use crate::syntax::r#abstract::{Term, WellTyped};
use crate::syntax::{Ctor, CtorParams, DataDef, FnDef, FnSig, Param, ID};

#[allow(dead_code)]
#[derive(Debug)]
enum Error<'src> {
    Mismatched { want: Term<'src>, got: Term<'src> },
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
    globals: HashMap<ID, WellTyped<'src>>,
    locals: HashMap<ID, Term<'src>>,
}

#[allow(dead_code)]
impl<'src> Checker<'src> {
    fn file(&mut self, f: File<'src>) -> Result<Box<[ID]>, Error<'src>> {
        f.decls
            .into_vec()
            .into_iter()
            .map(|d| self.decl(d))
            .collect()
    }

    fn decl(&mut self, decl: Decl<'src>) -> Result<ID, Error<'src>> {
        self.locals.clear();
        let id = decl.name.id;
        match decl.def {
            Def::Fn(d) => self.fn_def(id, d)?,
            Def::Data(d) => self.data_def(id, d)?,
        }
        Ok(id)
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
        let expected = ret.clone();
        self.globals.insert(
            id,
            WellTyped::UndefFn(FnSig {
                typ_params,
                val_params,
                eff,
                ret,
            }),
        );

        let body = self.check(&body, &expected)?;
        let sig = match self.globals.remove(&id).unwrap() {
            WellTyped::UndefFn(sig) => sig,
            _ => unreachable!(),
        };
        self.globals.insert(id, WellTyped::Fn(FnDef { sig, body }));

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
            .insert(id, WellTyped::Data(DataDef { typ_params, ctors }));
        Ok(())
    }

    fn ctor(
        &mut self,
        ctor: Ctor<'src, Expr<'src>>,
    ) -> Result<Ctor<'src, Term<'src>>, Error<'src>> {
        let Ctor { name, params } = ctor;
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
        p: Param<'src, Expr<'src>>,
    ) -> Result<Param<'src, Term<'src>>, Error<'src>> {
        let Param { name, typ } = p;
        let typ = self.check_type(&typ)?;
        self.locals.insert(name.id, *typ.clone());
        Ok(Param { name, typ })
    }

    fn check_type(&mut self, typ: &Expr<'src>) -> Result<Box<Term<'src>>, Error<'src>> {
        self.check(typ, &Term::Type)
    }

    fn check(&mut self, _e: &Expr<'src>, _ty: &Term<'src>) -> Result<Box<Term<'src>>, Error<'src>> {
        todo!()
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
            Expr::Fn { .. } => todo!(),
            Expr::Pure => Inferred::r#type(Term::Pure),
        })
    }
}
