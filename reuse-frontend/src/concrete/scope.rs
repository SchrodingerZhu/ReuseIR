use std::collections::HashMap;

use crate::concrete::{DeclExpr, Expr, FileExpr};
use crate::name::{Ident, ID};

#[allow(dead_code)]
enum Error<'src> {
    DuplicateName(&'src str),
    UnresolvedIdent(&'src str),
}

#[allow(dead_code)]
#[derive(Default)]
struct ScopeChecker<'src> {
    globals: HashMap<&'src str, ID>,
    locals: HashMap<&'src str, ID>,
}

impl<'src> ScopeChecker<'src> {
    #[allow(dead_code)]
    pub fn file(&mut self, file: &mut FileExpr<'src>) -> Result<(), Error<'src>> {
        file.decls
            .iter_mut()
            .try_fold((), |_, decl| self.decl(decl))
    }

    #[allow(dead_code)]
    fn decl(&mut self, _decl: &mut DeclExpr<'src>) -> Result<(), Error<'src>> {
        todo!()
    }

    #[allow(dead_code)]
    fn expr(&mut self, _expr: &mut Expr<'src>) -> Result<(), Error<'src>> {
        todo!()
    }

    #[allow(dead_code)]
    fn guarded<const N: usize>(
        &mut self,
        idents: [&Ident<'src>; N],
        expr: &mut Expr<'src>,
    ) -> Result<(), Error<'src>> {
        let olds = idents
            .into_iter()
            .filter_map(|&Ident { raw, id }| self.locals.insert(raw, id).map(|old| (raw, old)))
            .collect::<Vec<_>>();

        self.expr(expr)?;

        olds.into_iter().for_each(|(raw, id)| {
            self.locals.insert(raw, id);
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
    fn insert_global(&mut self, ident: Ident<'src>) -> Result<(), Error<'src>> {
        self.globals
            .insert(ident.raw, ident.id)
            .map_or(Ok(()), |_| Err(Error::DuplicateName(ident.raw)))
    }

    #[allow(dead_code)]
    fn clear_locals(&mut self) {
        self.locals.clear();
    }
}
