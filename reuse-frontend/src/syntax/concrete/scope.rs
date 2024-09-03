use crate::syntax::concrete::{Decl, Def, Expr, File};
use crate::syntax::{CtorParams, DataDef, FnDef, FnSig, Ident, NameMap, Param, ID};

#[allow(dead_code)]
#[derive(Debug)]
enum Error<'src> {
    DuplicateName(&'src str),
    UnresolvedIdent(&'src str),
}

#[allow(dead_code)]
#[derive(Default)]
struct Checker<'src> {
    globals: NameMap<'src>,
    locals: NameMap<'src>,
}

#[allow(dead_code)]
impl<'src> Checker<'src> {
    pub fn run(file: &mut File<'src>) {
        Self::default().file(file).unwrap();
    }

    fn file(&mut self, file: &mut File<'src>) -> Result<(), Error<'src>> {
        file.decls
            .iter()
            .try_fold((), |_, decl| self.insert_global(&decl.name))?;
        file.decls
            .iter_mut()
            .try_fold((), |_, decl| self.decl(decl))
    }

    fn decl(&mut self, decl: &mut Decl<'src>) -> Result<(), Error<'src>> {
        self.clear_locals();

        match &mut decl.def {
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
                typ_params.iter_mut().try_fold((), |_, p| self.param(p))?;
                val_params.iter_mut().try_fold((), |_, p| self.param(p))?;
                self.expr(eff)?;
                self.expr(ret)?;
                self.expr(body)
            }
            Def::Data(DataDef { typ_params, ctors }) => {
                typ_params.iter_mut().try_fold((), |_, p| self.param(p))?;
                ctors.iter_mut().try_fold((), |_, c| match &mut c.params {
                    CtorParams::None => Ok(()),
                    CtorParams::Unnamed(ts) => ts.iter_mut().try_fold((), |_, t| self.expr(t)),
                    CtorParams::Named(ps) => ps.iter_mut().try_fold((), |_, p| self.param(p)),
                })
            }
        }
    }

    fn expr(&mut self, expr: &mut Expr<'src>) -> Result<(), Error<'src>> {
        use Expr::*;
        match expr {
            PrimitiveType(..) | Primitive(..) => Ok(()),
            Ident(i) => self.ident(i),
            FnType {
                param_types,
                eff,
                ret,
            } => {
                param_types.iter_mut().try_fold((), |_, t| self.expr(t))?;
                self.expr(eff)?;
                self.expr(ret)
            }
            Fn { params, body } => self.with_idents(params, body),
            Call {
                f,
                typ_args,
                val_args,
            } => {
                self.expr(f)?;
                typ_args.iter_mut().try_fold((), |_, a| self.expr(a))?;
                val_args.iter_mut().try_fold((), |_, a| self.expr(a))
            }
            Let {
                name,
                typ,
                expr,
                body,
            } => {
                typ.as_mut().map(|t| self.expr(t)).transpose()?;
                self.expr(expr)?;
                self.with_ident(name, body)
            }
        }
    }

    fn with_ident(
        &mut self,
        ident: &Ident<'src>,
        expr: &mut Expr<'src>,
    ) -> Result<(), Error<'src>> {
        let old = self.insert_local(ident).map(|old| Ident {
            raw: ident.raw,
            id: old,
        });

        self.expr(expr)?;

        old.inspect(|o| {
            self.insert_local(o);
        });

        Ok(())
    }

    fn with_idents(
        &mut self,
        idents: &[Ident<'src>],
        expr: &mut Expr<'src>,
    ) -> Result<(), Error<'src>> {
        let olds = idents
            .iter()
            .filter_map(|i| {
                self.insert_local(i).map(|old| Ident {
                    raw: i.raw,
                    id: old,
                })
            })
            .collect::<Vec<_>>();

        self.expr(expr)?;

        olds.iter().for_each(|i| {
            self.insert_local(i);
        });

        Ok(())
    }

    fn ident(&mut self, ident: &mut Ident<'src>) -> Result<(), Error<'src>> {
        ident.id = *self
            .locals
            .get(ident.raw)
            .or_else(|| self.globals.get(ident.raw))
            .ok_or(Error::UnresolvedIdent(ident.raw))?;
        Ok(())
    }

    fn insert_global(&mut self, ident: &Ident<'src>) -> Result<(), Error<'src>> {
        self.globals
            .insert(ident.raw, ident.id)
            .map_or(Ok(()), |_| Err(Error::DuplicateName(ident.raw)))
    }

    fn insert_local(&mut self, ident: &Ident<'src>) -> Option<ID> {
        self.locals.insert(ident.raw, ident.id)
    }

    fn param(&mut self, param: &mut Param<'src, Expr<'src>>) -> Result<(), Error<'src>> {
        self.insert_local(&param.name);
        self.expr(&mut param.typ)
    }

    fn clear_locals(&mut self) {
        self.locals.clear();
    }
}

#[cfg(test)]
mod tests {
    use crate::syntax::concrete::{scope, Def, Expr, File};
    use crate::syntax::surface::parse;
    use crate::syntax::{FnDef, FnSig};

    fn check(src: &str) -> File {
        let mut file = parse(src);
        scope::Checker::run(&mut file);
        file
    }

    #[test]
    fn it_checks_scope() {
        check(
            "
def f0(): None
def f1(): f0()
        ",
        );
    }

    #[test]
    fn it_checks_lambda_scope() {
        match &check("def f(x: str): lambda x: x\n").decls[0].def {
            Def::Fn(FnDef {
                sig: FnSig { val_params, .. },
                body,
                ..
            }) => {
                let outer_id = val_params[0].name.id;
                match body.as_ref() {
                    Expr::Fn { params, body } => {
                        let inner_id = params[0].id;
                        match body.as_ref() {
                            Expr::Ident(i) => {
                                let id = i.id;
                                assert_eq!(id, inner_id);
                                assert_ne!(id, outer_id);
                            }
                            _ => assert!(false),
                        }
                    }
                    _ => assert!(false),
                }
            }
            _ => assert!(false),
        };
    }
}
