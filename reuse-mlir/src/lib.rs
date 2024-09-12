#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::{ffi::c_char, marker::PhantomData, ptr::NonNull};
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

type ContextToken<'ctx> = PhantomData<*mut &'ctx ()>;

macro_rules! wrapper {
    ($name:ident, $mlir_name:ty) => {
        #[derive(Clone, Copy)]
        #[repr(transparent)]
        pub struct $name<'ctx>($mlir_name, ContextToken<'ctx>);
    };
}

macro_rules! into_attr {
    ($from:ident) => {
        impl<'a> From<$from<'a>> for Attribute<'a> {
            fn from(attr: $from<'a>) -> Self {
                attr.0
            }
        }
    };
}

wrapper!(Context, MlirContext);
wrapper!(Module, MlirModule);
wrapper!(Location, MlirLocation);
wrapper!(StringRef, MlirStringRef);
wrapper!(Operation, MlirOperation);
wrapper!(Attribute, MlirAttribute);
wrapper!(Type, MlirType);
wrapper!(Value, MlirValue);
wrapper!(Block, MlirBlock);
wrapper!(Region, MlirRegion);
wrapper!(Function, Operation<'ctx>);
wrapper!(FunctionType, Type<'ctx>);
wrapper!(StringAttr, Attribute<'ctx>);
into_attr!(StringAttr);
wrapper!(IntegerAttr, Attribute<'ctx>);
into_attr!(IntegerAttr);
wrapper!(FlatSymbolRefAttr, Attribute<'ctx>);
into_attr!(FlatSymbolRefAttr);
wrapper!(IndexAttr, Attribute<'ctx>);
into_attr!(IndexAttr);
wrapper!(UnitAttr, Attribute<'ctx>);
into_attr!(UnitAttr);

impl<'a> StringRef<'a> {
    pub fn new(s: &str) -> Self {
        Self(
            MlirStringRef {
                data: s.as_ptr() as *const c_char,
                length: s.len(),
            },
            PhantomData,
        )
    }
}

impl<'a> From<&'a str> for StringRef<'a> {
    fn from(s: &'a str) -> Self {
        Self::new(s)
    }
}

impl<'a> Location<'a> {
    pub fn unknown(ctx: Context) -> Self {
        Self(unsafe { mlirLocationUnknownGet(ctx.0) }, PhantomData)
    }
    pub fn file_line_col(ctx: Context, filename: StringRef, line: u32, col: u32) -> Self {
        Self(
            unsafe { mlirLocationFileLineColGet(ctx.0, filename.0, line, col) },
            PhantomData,
        )
    }
}

impl Context<'_> {
    pub fn run<F>(f: F)
    where
        F: for<'a> FnOnce(Context<'a>),
    {
        unsafe {
            let handle = mlirContextCreate();
            let registry = mlirDialectRegistryCreate();
            let reuse_ir = mlirGetDialectHandle__reuse_ir__();
            mlirRegisterAllDialects(registry);
            mlirDialectHandleInsertDialect(reuse_ir, registry);
            mlirContextLoadAllAvailableDialects(handle);
            let context = Context(handle, PhantomData);
            f(context);
            mlirDialectRegistryDestroy(registry);
            mlirContextDestroy(handle)
        }
    }
}

impl<'a> Module<'a> {
    pub fn new(location: Location) -> Self {
        Self(unsafe { mlirModuleCreateEmpty(location.0) }, PhantomData)
    }
    pub fn set_name(&self, name: StringRef) {
        let op: Operation = (*self).into();
        op.set_attribute(
            "sym_name".into(),
            StringAttr::new(op.get_context(), name).into(),
        )
    }
}

impl<'a> From<Module<'a>> for Operation<'_> {
    fn from(module: Module) -> Self {
        unsafe {
            let op = mlirModuleGetOperation(module.0);
            Operation(op, PhantomData)
        }
    }
}

impl Operation<'_> {
    fn set_attribute(&self, name: StringRef, attr: Attribute) {
        unsafe { mlirOperationSetAttributeByName(self.0, name.0, attr.0) }
    }
    fn get_context(&self) -> Context {
        Context(unsafe { mlirOperationGetContext(self.0) }, PhantomData)
    }
}

struct MlirDisplayContext<'a> {
    f: NonNull<std::fmt::Formatter<'a>>,
    result: std::fmt::Result,
}

unsafe extern "C" fn mlir_display_callback(str_ref: MlirStringRef, ctx: *mut std::ffi::c_void) {
    let pctx = unsafe { &mut *(ctx as *mut MlirDisplayContext) };
    let slice = unsafe { std::slice::from_raw_parts(str_ref.data as *mut u8, str_ref.length) };
    let s = std::str::from_utf8(slice);
    pctx.result = pctx.result.and_then(|()| match s {
        Ok(s) => write!(unsafe { &mut *pctx.f.as_ptr() }, "{}", s),
        Err(_) => Err(std::fmt::Error),
    });
}

macro_rules! impl_display {
    ($ty:ident, $mlir_func:ident) => {
        impl<'ctx> std::fmt::Display for $ty<'ctx> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                let mut ctx = MlirDisplayContext {
                    f: NonNull::from(f),
                    result: Ok(()),
                };
                unsafe {
                    $mlir_func(
                        self.0,
                        Some(mlir_display_callback),
                        &mut ctx as *mut _ as *mut std::ffi::c_void,
                    );
                }
                ctx.result
            }
        }
    };
}

impl_display!(Operation, mlirOperationPrint);
impl_display!(Type, mlirTypePrint);
impl_display!(Attribute, mlirAttributePrint);

impl Function<'_> {
    pub fn new(_name: StringRef, location: Location) -> Self {
        unsafe {
            let func_op = StringRef::new("func.func");
            let _state = mlirOperationStateGet(func_op.0, location.0);
        }
        unimplemented!()
    }
}

impl FlatSymbolRefAttr<'_> {
    pub fn new(ctx: Context, symbol: StringRef) -> Self {
        Self(
            unsafe { Attribute(mlirFlatSymbolRefAttrGet(ctx.0, symbol.0), PhantomData) },
            PhantomData,
        )
    }
}

impl StringAttr<'_> {
    pub fn new(ctx: Context, value: StringRef) -> Self {
        Self(
            unsafe { Attribute(mlirStringAttrGet(ctx.0, value.0), PhantomData) },
            PhantomData,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_gets_reuse_ir_dialect_handle() {
        unsafe {
            assert!(!mlirGetDialectHandle__reuse_ir__().ptr.is_null());
        }
    }

    #[test]
    fn it_creates_empty_module() {
        Context::run(|ctx| {
            let location = Location::file_line_col(ctx, "test.mlir".into(), 0, 0);
            let module = Module::new(location);
            module.set_name("test".into());
            let operation: Operation = module.into();
            println!("{}", operation);
        });
    }
}
