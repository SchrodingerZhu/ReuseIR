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
        both!(None) => unreachable!(),
        both!(Boolean) => true,
        both!(False) => unreachable!(),
        both!(True) => unreachable!(),
        both!(String) => true,
        both!(Str(..)) => unreachable!(),
        both!(F32) => true,
        both!(F64) => true,
        both!(Float(..)) => unreachable!(),
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
        both!(Pure) => true,
        (GenericFnType { .. }, GenericFnType { .. }) => todo!(),
        (GenericFn { .. }, GenericFn { .. }) => todo!(),

        _ => false,
    }
}
