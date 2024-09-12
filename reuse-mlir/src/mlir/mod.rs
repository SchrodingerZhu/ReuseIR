#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

mod ffi;

use ffi::*;
use std::{cell::UnsafeCell, ffi::c_char, marker::PhantomData, ptr::NonNull};

type ContextToken<'ctx> = PhantomData<*mut &'ctx ()>;

macro_rules! wrapper {
    ($name:ident, $mlir_name:ty) => {
        #[repr(transparent)]
        pub struct $name<'ctx>($mlir_name, ContextToken<'ctx>);
    };
}

macro_rules! impl_from {
    ($from:ident, $to:ident) => {
        impl<'a> From<$from<'a>> for $to<'a> {
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
wrapper!(Attribute, MlirAttribute);
wrapper!(Type, MlirType);
wrapper!(Value, MlirValue);
wrapper!(Operation, MlirOperation);
wrapper!(Region, MlirRegion);
wrapper!(Block, MlirBlock);
wrapper!(Function, Operation<'ctx>);
impl_from!(Function, Operation);
wrapper!(FunctionType, Type<'ctx>);
impl_from!(FunctionType, Type);
wrapper!(StringAttr, Attribute<'ctx>);
impl_from!(StringAttr, Attribute);
wrapper!(IntegerAttr, Attribute<'ctx>);
impl_from!(IntegerAttr, Attribute);
wrapper!(FlatSymbolRefAttr, Attribute<'ctx>);
impl_from!(FlatSymbolRefAttr, Attribute);
wrapper!(IndexAttr, Attribute<'ctx>);
impl_from!(IndexAttr, Attribute);
wrapper!(UnitAttr, Attribute<'ctx>);
impl_from!(UnitAttr, Attribute);
wrapper!(TypeAttr, Attribute<'ctx>);
impl_from!(TypeAttr, Attribute);
wrapper!(Identifer, MlirIdentifier);
wrapper!(NamedAttribute, MlirNamedAttribute);

impl Copy for Context<'_> {}
impl Clone for Context<'_> {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for Location<'_> {}
impl Clone for Location<'_> {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for Attribute<'_> {}
impl Clone for Attribute<'_> {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for Type<'_> {}
impl Clone for Type<'_> {
    fn clone(&self) -> Self {
        *self
    }
}

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
        let op = Operation(unsafe { mlirModuleGetOperation(self.0) }, self.1);
        op.set_attribute("sym_name", StringAttr::new(op.get_context(), name));
        std::mem::forget(op);
    }
    pub fn body<F>(&self, f: F)
    where
        F: FnOnce(&Block<'a>),
    {
        let handle = unsafe { mlirModuleGetBody(self.0) };
        let block = Block(handle, self.1);
        f(&block);
        std::mem::forget(block);
    }
}

impl<'a> Operation<'a> {
    fn set_attribute<S, A>(&self, name: S, attr: A)
    where
        S: Into<StringRef<'a>>,
        A: Into<Attribute<'a>>,
    {
        unsafe { mlirOperationSetAttributeByName(self.0, name.into().0, attr.into().0) }
    }
    fn get_context(&self) -> Context<'a> {
        Context(unsafe { mlirOperationGetContext(self.0) }, self.1)
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
impl_display!(Value, mlirValuePrint);

pub enum Visibility {
    Public,
    Private,
    Nested,
}

pub enum LLVMLinkage {
    External,
    AvailableExternally,
    Linkonce,
    LinkonceODR,
    Weak,
    WeakODR,
    Appending,
    Internal,
    Private,
}

#[repr(transparent)]
pub struct OperationBuilder<'ctx>(UnsafeCell<MlirOperationState>, ContextToken<'ctx>);

impl<'ctx> OperationBuilder<'ctx> {
    pub fn new<'a, N: Into<StringRef<'a>>>(name: N, location: Location) -> Self {
        let state = unsafe { mlirOperationStateGet(name.into().0, location.0) };
        Self(UnsafeCell::new(state), PhantomData)
    }
    pub fn add_attributes(self, named_attrs: &[NamedAttribute]) -> Self {
        unsafe {
            mlirOperationStateAddAttributes(
                self.0.get(),
                named_attrs.len() as _,
                named_attrs.as_ptr() as _,
            )
        }
        self
    }
    pub fn add_attribute(self, named_attr: NamedAttribute) -> Self {
        self.add_attributes(&[named_attr])
    }
    pub fn add_operands(self, values: &[Value]) -> Self {
        unsafe {
            mlirOperationStateAddOperands(self.0.get(), values.len() as _, values.as_ptr() as _)
        }
        self
    }
    pub fn add_operand(self, value: Value) -> Self {
        self.add_operands(&[value])
    }
    pub fn add_results(self, values: &[Type]) -> Self {
        unsafe {
            mlirOperationStateAddResults(self.0.get(), values.len() as _, values.as_ptr() as _)
        }
        self
    }
    pub fn add_result(self, value: Type) -> Self {
        self.add_results(&[value])
    }
    pub fn add_regions<const N: usize>(self, regions: [Region; N]) -> Self {
        unsafe {
            mlirOperationStateAddOwnedRegions(
                self.0.get(),
                regions.len() as _,
                regions.as_ptr() as _,
            )
        }
        std::mem::forget(regions);
        self
    }
    pub fn add_region(self, region: Region) -> Self {
        self.add_regions([region])
    }
    pub fn build(self) -> Operation<'ctx> {
        let op = unsafe { mlirOperationCreate(self.0.get()) };
        Operation(op, PhantomData)
    }
}

impl Function<'_> {
    pub fn new(
        name: StringRef,
        location: Location,
        r#type: FunctionType,
        visibility: Visibility,
        region: Region,
    ) -> Self {
        let ctx = Context(unsafe { mlirLocationGetContext(location.0) }, location.1);
        let op = OperationBuilder::new("func.func", location)
            .add_attributes(&[
                NamedAttribute::new(Identifer::new(ctx, "sym_name"), StringAttr::new(ctx, name)),
                NamedAttribute::new(
                    Identifer::new(ctx, "function_type"),
                    TypeAttr::new(r#type.into()),
                ),
                NamedAttribute::new(
                    Identifer::new(ctx, "sym_visibility"),
                    StringAttr::new(
                        ctx,
                        match visibility {
                            Visibility::Public => "public",
                            Visibility::Private => "private",
                            Visibility::Nested => "nested",
                        },
                    ),
                ),
            ])
            .add_region(region)
            .build();
        Self(op, PhantomData)
    }

    pub fn set_linkage(&self, linkage: LLVMLinkage) {
        let linkage = match linkage {
            LLVMLinkage::External => MlirLLVMLinkage_MlirLLVMLinkageExternal,
            LLVMLinkage::AvailableExternally => MlirLLVMLinkage_MlirLLVMLinkageAvailableExternally,
            LLVMLinkage::Linkonce => MlirLLVMLinkage_MlirLLVMLinkageLinkonce,
            LLVMLinkage::LinkonceODR => MlirLLVMLinkage_MlirLLVMLinkageLinkonceODR,
            LLVMLinkage::Weak => MlirLLVMLinkage_MlirLLVMLinkageWeak,
            LLVMLinkage::WeakODR => MlirLLVMLinkage_MlirLLVMLinkageWeakODR,
            LLVMLinkage::Appending => MlirLLVMLinkage_MlirLLVMLinkageAppending,
            LLVMLinkage::Internal => MlirLLVMLinkage_MlirLLVMLinkageInternal,
            LLVMLinkage::Private => MlirLLVMLinkage_MlirLLVMLinkagePrivate,
        };
        let attr = unsafe {
            let ctx = self.0.get_context();
            Attribute(mlirLLVMLinkageAttrGet(ctx.0, linkage), ctx.1)
        };
        self.0.set_attribute("llvm.linkage", attr);
    }
}

impl TypeAttr<'_> {
    pub fn new(ty: Type) -> Self {
        Self(
            unsafe { Attribute(mlirTypeAttrGet(ty.0), PhantomData) },
            PhantomData,
        )
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
    pub fn new<'a, S: Into<StringRef<'a>>>(ctx: Context, value: S) -> Self {
        Self(
            unsafe { Attribute(mlirStringAttrGet(ctx.0, value.into().0), PhantomData) },
            PhantomData,
        )
    }
}

impl FunctionType<'_> {
    pub fn new(ctx: Context, inputs: &[Type], results: &[Type]) -> Self {
        let handle = unsafe {
            mlirFunctionTypeGet(
                ctx.0,
                inputs.len() as _,
                inputs.as_ptr() as _,
                results.len() as _,
                results.as_ptr() as _,
            )
        };
        Self(Type(handle, PhantomData), PhantomData)
    }
}

impl<'a> Identifer<'a> {
    pub fn new<'b, S: Into<StringRef<'b>>>(ctx: Context, name: S) -> Self {
        Self(
            unsafe { mlirIdentifierGet(ctx.0, name.into().0) },
            PhantomData,
        )
    }
    pub fn as_string(&self) -> StringRef<'a> {
        let str_ref = unsafe { mlirIdentifierStr(self.0) };
        StringRef(str_ref, PhantomData)
    }
}

impl<'a> NamedAttribute<'a> {
    pub fn new<F: Into<Attribute<'a>>>(name: Identifer, attr: F) -> Self {
        Self(
            unsafe { mlirNamedAttributeGet(name.0, attr.into().0) },
            PhantomData,
        )
    }
}

impl<'a> Region<'a> {
    pub fn new(_ctx: Context<'a>) -> Self {
        Self(unsafe { mlirRegionCreate() }, _ctx.1)
    }
    pub fn append_block(&self, block: Block) {
        let handle = block.0;
        std::mem::forget(block);
        unsafe { mlirRegionAppendOwnedBlock(self.0, handle) }
    }
}

impl<'a> Block<'a> {
    pub fn new(_ctx: Context<'a>, types: &[Type], locs: &[Location]) -> Self {
        let min = types.len().min(locs.len());
        Self(
            unsafe { mlirBlockCreate(min as _, types.as_ptr() as _, locs.as_ptr() as _) },
            _ctx.1,
        )
    }
    pub fn append_operation<O>(&self, op: O)
    where
        O: Into<Operation<'a>>,
    {
        let op = op.into();
        unsafe { mlirBlockAppendOwnedOperation(self.0, op.0) };
        std::mem::forget(op);
    }
}

impl Drop for Region<'_> {
    fn drop(&mut self) {
        unsafe { mlirRegionDestroy(self.0) }
    }
}

impl Drop for Block<'_> {
    fn drop(&mut self) {
        unsafe { mlirBlockDestroy(self.0) }
    }
}

impl Drop for Operation<'_> {
    fn drop(&mut self) {
        unsafe { mlirOperationDestroy(self.0) }
    }
}
impl Drop for Module<'_> {
    fn drop(&mut self) {
        unsafe { mlirModuleDestroy(self.0) }
    }
}
impl std::fmt::Display for Module<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let op = Operation(unsafe { mlirModuleGetOperation(self.0) }, self.1);
        let res = op.fmt(f);
        std::mem::forget(op);
        res
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
            println!("{}", module);
        });
    }

    #[test]
    fn it_creates_function() {
        Context::run(|ctx| {
            let location = Location::file_line_col(ctx, "test.mlir".into(), 0, 0);
            let module = Module::new(location);
            module.set_name("test".into());
            let function_type = FunctionType::new(ctx, &[], &[]);
            let region = Region::new(ctx);
            let block = Block::new(ctx, &[], &[]);
            let ret = OperationBuilder::new("func.return", location).build();
            block.append_operation(ret);
            region.append_block(block);
            let function = Function::new(
                "test".into(),
                location,
                function_type,
                Visibility::Private,
                region,
            );
            module.body(move |block| block.append_operation(function));
            println!("{module}");
        });
    }
}
