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
        both!(Primitive(..)) => unreachable!(),
        (PrimitiveType(a), PrimitiveType(b)) => a == b,
        (Ident(a), Ident(b)) => a.id == b.id,
        (
            FnType {
                param_types: param_types_lhs,
                eff: eff_lhs,
                ret: ret_lhs,
            },
            FnType {
                param_types: param_types_rhs,
                eff: eff_rhs,
                ret: ret_rhs,
            },
        ) if param_types_lhs.len() == param_types_rhs.len() => {
            param_types_lhs
                .iter()
                .zip(param_types_rhs)
                .all(|(l, r)| convert(l, r))
                && convert(eff_lhs, eff_rhs)
                && convert(ret_lhs, ret_rhs)
        }
        both!(Fn { .. }) => unreachable!(),
        both!(Call { .. }) => unreachable!(),
        (GenericFnType { .. }, GenericFnType { .. }) => todo!(),
        (GenericFn { .. }, GenericFn { .. }) => todo!(),
        _ => false,
    }
}
