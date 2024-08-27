use std::collections::HashMap;

use crate::syntax::concrete::{DeclExpr, Expr, FileExpr, ParamExpr};
use crate::syntax::{CtorParams, DataDef, Def, FnDef, Ident, ID};

#[allow(dead_code)]
enum Error<'src> {
    DuplicateName(&'src str),
    UnresolvedIdent(&'src str),
}

type NameMap<'src> = HashMap<&'src str, ID>;

#[allow(dead_code)]
#[derive(Default)]
struct ScopeChecker<'src> {
    globals: NameMap<'src>,
    locals: NameMap<'src>,
}

impl<'src> ScopeChecker<'src> {
    #[allow(dead_code)]
    pub fn file(&mut self, file: &mut FileExpr<'src>) -> Result<(), Error<'src>> {
        file.decls
            .iter()
            .try_fold((), |_, decl| self.insert_global(&decl.name))?;
        file.decls
            .iter_mut()
            .try_fold((), |_, decl| self.decl(decl))
    }

    #[allow(dead_code)]
    fn decl(&mut self, decl: &mut DeclExpr<'src>) -> Result<(), Error<'src>> {
        self.clear_locals();

        match &mut decl.def {
            Def::Fn(FnDef {
                typ_params,
                val_params,
                eff,
                ret,
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

    #[allow(dead_code)]
    fn expr(&mut self, expr: &mut Expr<'src>) -> Result<(), Error<'src>> {
        match expr {
            Expr::Ident(i) => self.ident(i),

            Expr::FnType {
                param_types,
                eff,
                ret,
            } => {
                param_types.iter_mut().try_fold((), |_, t| self.expr(t))?;
                self.expr(eff)?;
                self.expr(ret)
            }
            Expr::Fn { params, body } => self.guarded(params, body),

            Expr::Type
            | Expr::NoneType
            | Expr::None
            | Expr::Boolean
            | Expr::False
            | Expr::True
            | Expr::String
            | Expr::Str(..)
            | Expr::F32
            | Expr::F64
            | Expr::Float(..)
            | Expr::Pure => Ok(()),
        }
    }

    #[allow(dead_code)]
    fn guarded(
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

    #[allow(dead_code)]
    fn ident(&mut self, ident: &mut Ident<'src>) -> Result<(), Error<'src>> {
        ident.id = *self
            .locals
            .get(ident.raw)
            .or_else(|| self.globals.get(ident.raw))
            .ok_or(Error::UnresolvedIdent(ident.raw))?;
        Ok(())
    }

    #[allow(dead_code)]
    fn insert_global(&mut self, ident: &Ident<'src>) -> Result<(), Error<'src>> {
        self.globals
            .insert(ident.raw, ident.id)
            .map_or(Ok(()), |_| Err(Error::DuplicateName(ident.raw)))
    }

    #[allow(dead_code)]
    fn insert_local(&mut self, ident: &Ident<'src>) -> Option<ID> {
        self.locals.insert(ident.raw, ident.id)
    }

    #[allow(dead_code)]
    fn param(&mut self, param: &mut ParamExpr<'src>) -> Result<(), Error<'src>> {
        self.insert_local(&param.name);
        self.expr(&mut param.typ)
    }

    #[allow(dead_code)]
    fn clear_locals(&mut self) {
        self.locals.clear();
    }
}
