use crate::name::Name;

trait Syntax {}

#[allow(dead_code)]
struct Param<T: Syntax> {
    name: Name,
    typ: Box<T>,
}

#[allow(dead_code)]
struct Decl<T: Syntax> {
    name: Name,
    typ_params: Box<[Param<T>]>,
    val_params: Box<[Param<T>]>,
    eff: Box<T>,
    ret: Box<T>,
    def: Def<T>,
}

#[allow(dead_code)]
enum Def<T: Syntax> {
    Fn(FnDef<T>),
    Data(DataDef<T>),
}

#[allow(dead_code)]
struct FnDef<T: Syntax> {
    f: Box<T>,
}

#[allow(dead_code)]
struct DataDef<T: Syntax> {
    ctors: Box<[Ctor<T>]>,
}

#[allow(dead_code)]
struct Ctor<T: Syntax> {
    name: Name,
    val_params: Box<[Param<T>]>,
}
