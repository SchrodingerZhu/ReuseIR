use std::collections::HashMap;

use crate::syntax::r#abstract::Term;
use crate::syntax::ID;

#[allow(dead_code)]
#[derive(Default)]
pub struct Inliner<'src> {
    env: HashMap<ID, Term<'src>>,
}

#[allow(dead_code)]
impl<'src> Inliner<'src> {
    pub fn with(mut self, id: ID, typ: Term<'src>) -> Self {
        self.env.insert(id, typ);
        self
    }

    pub fn term(&mut self, tm: Term<'src>) -> Term<'src> {
        use Term::*;
        match tm {
            Ident(i) => self.env.get(&i.id).cloned().unwrap_or(Ident(i)),
            Type => Type,
            NoneType => NoneType,
            None => None,
            Boolean => Boolean,
            False => False,
            True => True,
            String => String,
            Str(s) => Str(s),
            F32 => F32,
            F64 => F64,
            Float(v) => Float(v),
            FnType {
                param_types,
                eff,
                ret,
            } => FnType {
                param_types: param_types
                    .into_vec()
                    .into_iter()
                    .map(|t| Box::new(self.term(*t)))
                    .collect(),
                eff: Box::new(self.term(*eff)),
                ret: Box::new(self.term(*ret)),
            },
            Fn { params, body } => Fn {
                params,
                body: Box::new(self.term(*body)),
            },
            Pure => Pure,
            GenericFnType { mut param, body } => {
                param.typ = Box::new(self.term(*param.typ));
                let body = Box::new(self.term(*body));
                GenericFnType { param, body }
            }
            GenericFn { mut param, body } => {
                param.typ = Box::new(self.term(*param.typ));
                let body = Box::new(self.term(*body));
                GenericFn { param, body }
            }
        }
    }
}
