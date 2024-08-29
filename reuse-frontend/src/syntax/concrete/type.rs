use std::collections::HashMap;

use crate::syntax::concrete::{Decl, Def, Expr, File};
use crate::syntax::r#abstract::{Term, WellTyped};
use crate::syntax::{FnDef, FnSig, Param, ID};

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
            Def::Fn(FnDef {
                sig:
                    FnSig {
                        typ_params,
                        val_params,
                        eff,
                        ret,
                    },
                body,
            }) => {
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
            }
            Def::Data(_) => todo!(),
        }

        Ok(id)
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

    fn infer(&mut self, _e: &Expr<'src>) -> Result<Box<Inferred<'src>>, Error<'src>> {
        todo!()
    }
}
