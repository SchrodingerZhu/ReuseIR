#ifndef REUSE_IR_TYPES
#define REUSE_IR_TYPES

include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/EnumAttr.td"
include "ReuseIR/IR/ReuseIRDialect.td"
include "ReuseIR/Interfaces/ReuseIRMangleInterface.td"
include "ReuseIR/Interfaces/ReuseIRCompositeLayoutInterface.td"

def AK_NonAtomic : I32EnumAttrCase<"nonatomic", 0>;
def AK_Atomic : I32EnumAttrCase<"atomic", 1>;

def FK_Nonfreezing : I32EnumAttrCase<"nonfreezing", 0>;
def FK_Frozen : I32EnumAttrCase<"frozen", 1>;
def FK_Unfrozen : I32EnumAttrCase<"unfrozen", 2>;

def AtomicKind : I32EnumAttr<
    "AtomicKind",
    "atomic kind",
    [AK_NonAtomic, AK_Atomic]> {
  let cppNamespace = "::mlir::reuse_ir";
}

def FreezingKind : I32EnumAttr<
    "FreezingKind",
    "freezing kind",
    [FK_Nonfreezing, FK_Frozen, FK_Unfrozen]> {
  let cppNamespace = "::mlir::reuse_ir";
}

class ReuseIR_Type<string name, string typeMnemonic, list<Trait> traits = [],
               string baseCppClass = "::mlir::Type">
    : TypeDef<ReuseIR_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

def ReuseIR_IntType : AnyTypeOf<[
    Builtin_Index, Builtin_Integer
]>;

def ReuseIR_FPType : AnyTypeOf<[
    Builtin_Float16, Builtin_Float32, Builtin_Float64, Builtin_Float128
]>;

def ReuseIR_TokenType : ReuseIR_Type<"Token", "token",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>, DeclareTypeInterfaceMethods<ReuseIRMangleInterface>]> {
  let summary = "ReuseIR Memory Token";
  let description = [{
    `reuse_ir.token` is the token of the memory
  }];
  let parameters = (ins
    "size_t":$alignment,
    "size_t":$size
  );
  let assemblyFormat = [{
    `<` `size` `:` $size `,` `alignment` `:` $alignment `>`
  }];
  let genVerifyDecl = 1;
}

def ReuseIR_RegionCtxType : ReuseIR_Type<"RegionCtx", "region_ctx",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {
  let summary = "ReuseIR Region Context";
  let description = [{
    `reuse_ir.region_ctx` is used to track mutable subregions.
    It contains the root for regional garbage collection.
    Region {
      FreezableRcHeader* tail;
    }
    RegionCtx is a pointer to Region. For now, it can be understood as a pointer to a pointer.
  }];
}

def ReuseIR_NullableType : ReuseIR_Type<"Nullable", "nullable",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {
  let summary = "ReuseIR Nullable Type";
  let description = [{
    `reuse_ir.nullable` is a nullable type. Rc pointers and tokens on default at nonnull in ReuseIR.
    This is just a type wrapper to indicate that the pointer can be null.
  }];
  let parameters = (ins
    "::mlir::Type":$pointer
  );
  let assemblyFormat = [{
    `<` $pointer `>`
  }];
}

def ReuseIR_RcBoxType : ReuseIR_Type<"RcBox", "rcbox",
    [
      DeclareTypeInterfaceMethods<DataLayoutTypeInterface>,
      DeclareTypeInterfaceMethods<ReuseIRCompositeLayoutInterface>
    ]> {
  let summary = "ReuseIR Control Box";
  let description = [{
    `reuse_ir.rcbox` is the underlying control box of RC managed objects.
    It has the following varaints:
    - Non-atomic plain RC
    - Non-atomic freezable RC
    - Atomic plain RC
    - Atomic freezable RC
    For Non-freezable variants, status is simply an index type:
    ```
    RcBox<T> {
        size_t status;
        T data;
    };
    ```
    For freezable varaints, status is also of the same size but it is actually 
    a tagged pointer. Please refer to https://dl.acm.org/doi/abs/10.1145/3652024.3665507
    for more details:
    ```
    RcBox<T> {
        union {
          size_t status;
          RcBox* root;
        } status;
        RcBox* next;
        void* vtable;
        T data;
    };
    struct VTable {
      void*  drop;
      size_t size;
      size_t alignment;
      ssize_t scan_count;
      size_t scan_offset[0];
    }
    - if scan_count >= 0, then scan_offset stores offsets for inner mutable fields (composite types).;
    - if scan_count < 0, then scan_offset contains a single value annotates the stride between inner mutable fields (array types).
    ```
  }];
  let parameters = (ins
    "mlir::Type":$dataType,
    "::mlir::reuse_ir::AtomicKindAttr":$atomicKind,
    "::mlir::reuse_ir::FreezingKindAttr":$freezingKind
  );
  let assemblyFormat = [{
    `<` $dataType `,` custom<AtomicKind>($atomicKind) `,` custom<FreezingKind>($freezingKind) `>`
  }];
}

def ReuseIR_RcType : ReuseIR_Type<"Rc", "rc",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>, DeclareTypeInterfaceMethods<ReuseIRMangleInterface>]> {

  let summary = "ReuseIR Rc Pointer Type";
  let description = [{
    `reuse_ir.rc` is the Rc pointer to managed object (in box)
  }];

  let parameters = (ins
    "mlir::Type":$pointee,
    "::mlir::reuse_ir::AtomicKindAttr":$atomicKind,
    "::mlir::reuse_ir::FreezingKindAttr":$freezingKind
  );

  let assemblyFormat = [{
    `<` $pointee `,` custom<AtomicKind>($atomicKind) `,` custom<FreezingKind>($freezingKind) `>`
  }];
}

def ReuseIR_MRefType : ReuseIR_Type<"MRef", "mref",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {

  let summary = "ReuseIR Mutable Reference Type";
  let description = [{
    `reuse_ir.mref` is the mutable reference that is assignable inside a unfrozen but freezable object.
  }];

  let parameters = (ins
    "mlir::Type":$pointee,
    "::mlir::reuse_ir::AtomicKindAttr":$atomicKind
  );

  let assemblyFormat = [{
    `<` $pointee `,` custom<AtomicKind>($atomicKind) `>`
  }];
}

def ReuseIR_RefType : ReuseIR_Type<"Ref", "ref",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {

  let summary = "ReuseIR Reference Type";
  let description = [{
    `reuse_ir.ref` is the referece to the underlying type. It serves as the pointer
    in ReuseIR.
  }];

  let parameters = (ins
    "mlir::Type":$pointee,
    "::mlir::reuse_ir::FreezingKindAttr":$freezingKind
  );

  let assemblyFormat = [{
    `<` $pointee `,` custom<FreezingKind>($freezingKind) `>`
  }];
}

def ReuseIR_ArrayType : ReuseIR_Type<"Array", "array",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>, DeclareTypeInterfaceMethods<ReuseIRMangleInterface>]> {

  let summary = "ReuseIR Array Type";
  let description = [{
    `reuse_ir.array` is a multi-dimensional array with fixed lengths
  }];

  let parameters = (ins
    "mlir::Type":$elementType,
    ArrayRefParameter<"size_t">:$sizes
  );

  let assemblyFormat = [{
    `<` $elementType `,` $sizes `>`
  }];
}

def ReuseIR_VectorType : ReuseIR_Type<"Vector", "vector",
    [
      DeclareTypeInterfaceMethods<DataLayoutTypeInterface>,
      DeclareTypeInterfaceMethods<ReuseIRCompositeLayoutInterface>,
      DeclareTypeInterfaceMethods<ReuseIRMangleInterface>
    ]> {
  let summary = "ReuseIR Vector Type";
  let description = [{
    `reuse_ir.vector` is a single-dimension vector with dynamic length.
    Vec<T> {
      T* data;
      size_t capacity;
      size_t length;
    }
  }];

  let parameters = (ins
    "mlir::Type":$elementType
  );

  let assemblyFormat = [{
    `<` $elementType `>`
  }];
}

def ReuseIR_OpaqueType : ReuseIR_Type<"Opaque", "opaque",
    [
      DeclareTypeInterfaceMethods<DataLayoutTypeInterface>,
      DeclareTypeInterfaceMethods<ReuseIRCompositeLayoutInterface>,
      DeclareTypeInterfaceMethods<ReuseIRMangleInterface>
    ]> {
  let summary = "ReuseIR Opaque Type";
  let description = [{
    `reuse_ir.opaque` is an opaque type for externally managed object.
    Opaque {
      void* drop;
      void* clone;
      T data;
    }
  }];

  let parameters = (ins
    "mlir::IntegerAttr":$size,
    "mlir::IntegerAttr":$alignment
  );

  let assemblyFormat = [{
    `<` $size `,` $alignment `>`
  }];
}


def ReuseIR_CompositeType : Type<CPred<"::mlir::isa<::mlir::reuse_ir::CompositeType>($_self)">,
                          "ReuseIR composite type">;

def ReuseIR_UnionType : Type<CPred<"::mlir::isa<::mlir::reuse_ir::UnionType>($_self)">,
                          "ReuseIR union  type">;

def ReuseIR_ClosureType : ReuseIR_Type<"Closure", "closure",
    [
      DeclareTypeInterfaceMethods<DataLayoutTypeInterface>, 
      DeclareTypeInterfaceMethods<ReuseIRCompositeLayoutInterface>,
      DeclareTypeInterfaceMethods<ReuseIRMangleInterface>
    ]> {

  let summary = "ReuseIR Closure Type";
  let description = [{
    `reuse_ir.closure` is for capturing lambda expressions.
    struct Closure {
      void* vtable;
      void* arg_start;
      void* arg_cursor;
    };
    struct VTable {
      void* func;
      void* clone;
      void* drop;
    }
  }];

  let parameters = (ins
    ArrayRefParameter<"::mlir::Type">:$inputTypes,
    OptionalParameter<"::mlir::Type">:$outputType
  );

  let assemblyFormat = [{
    `<` `(` $inputTypes `)` (`->` $outputType^)? `>`
  }];
}

class ReuseIR_RcTypeOf<list<Type> allowedTypes> : Type<
  /*predicate*/ Concat<"[](::mlir::Type pointee) { return ",
         SubstLeaves<"$_self", "pointee", AnyTypeOf<allowedTypes>.predicate>,
         "; }(::llvm::cast<::mlir::reuse_ir::RcType>($_self).getPointee())">,
  /* descr */ "RcType of certain types",
  /* baseType */ "::mlir::reuse_ir::RcType"
>;

class ReuseIR_RefTypeOf<list<Type> allowedTypes> : Type<
  /*predicate*/ Concat<"[](::mlir::Type pointee) { return ",
         SubstLeaves<"$_self", "pointee", AnyTypeOf<allowedTypes>.predicate>,
         "; }(::llvm::cast<::mlir::reuse_ir::RefType>($_self).getPointee())">,
  /* descr */ "RcType of certain types",
  /* baseType */ "::mlir::reuse_ir::RefType"
>;

def ReuseIR_NullableToken : Type<
  /*predicate*/ Concat<"[](::mlir::Type pointer) { return ",
         SubstLeaves<"$_self", "pointer", AnyTypeOf<[ReuseIR_TokenType]>.predicate>,
         "; }(::llvm::cast<::mlir::reuse_ir::NullableType>($_self).getPointer())">,
  /* descr */ "RcType of certain types",
  /* baseType */ "::mlir::reuse_ir::NullableType"
>;

def ReuseIR_PtrType : ReuseIR_Type<"Ptr", "ptr",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {
  let summary = "ReuseIR opaque pointer type";
  let description = [{
    `reuse_ir.ptr` is an opaque pointer type.
  }];
}

#endif // REUSE_IR_TYPES
