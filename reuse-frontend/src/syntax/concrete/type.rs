use std::collections::HashMap;

use crate::syntax::r#abstract::{Decl as WellTyped, Term};
use crate::syntax::ID;

#[allow(dead_code)]
#[derive(Default)]
struct Checker<'src> {
    globals: HashMap<ID, WellTyped<'src>>,
    locals: HashMap<ID, Term<'src>>,
}
