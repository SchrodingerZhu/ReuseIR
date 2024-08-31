use crate::syntax::r#abstract::Term;

macro_rules! both {
    ($t:pat) => {
        ($t, $t)
    };
}

#[allow(dead_code)]
pub fn convert<'src>(want: &'src Term<'src>, got: &'src Term<'src>) -> bool {
    use Term::*;
    match (want, got) {
        (Ident(a), Ident(b)) => a.id == b.id,
        both!(Type) => true,
        both!(NoneType) => true,
        both!(None) => true,
        both!(Boolean) => true,
        both!(False) => true,
        both!(True) => true,
        both!(String) => true,
        (Str(a), Str(b)) => a == b,
        both!(F32) => true,
        both!(F64) => true,
        (Float(a), Float(b)) => a == b,
        (FnType { .. }, FnType { .. }) => todo!(),
        (Fn { .. }, Fn { .. }) => todo!(),
        both!(Pure) => true,
        (GenericFnType { .. }, GenericFnType { .. }) => todo!(),
        (GenericFn { .. }, GenericFn { .. }) => todo!(),

        _ => false,
    }
}
