#ifndef REUSE_IR_TYPES
#define REUSE_IR_TYPES

include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/IR/BuiltinTypes.td"
include "ReuseIR/IR/ReuseIRDialect.td"
include "ReuseIR/Interfaces/ReuseIRMangleInterface.td"
include "ReuseIR/Interfaces/ReuseIRCompositeLayoutInterface.td"

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
  }];
}

def ReuseIR_RcBoxType : ReuseIR_Type<"RcBox", "rcbox",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>,DeclareTypeInterfaceMethods<ReuseIRCompositeLayoutInterface>]> {
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
      size_t scan_count;
      size_t scan_offset[0];
    }
    ```
  }];
  let parameters = (ins
    "mlir::Type":$dataType,
    OptionalParameter<"mlir::BoolAttr">:$atomic,
    OptionalParameter<"mlir::BoolAttr">:$freezable
  );
  let assemblyFormat = [{
    `<` $dataType (`,` `atomic` `:` $atomic^ )? (`,` `freezable` `:` $freezable^ )? `>`
  }];
}

def ReuseIR_RcType : ReuseIR_Type<"Rc", "rc",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {

  let summary = "ReuseIR Rc Pointer Type";
  let description = [{
    `reuse_ir.rc` is the Rc pointer to managed object (in box)
  }];

  let parameters = (ins
    "mlir::Type":$pointee,
    OptionalParameter<"mlir::BoolAttr">:$atomic,
    OptionalParameter<"mlir::BoolAttr">:$frozen
  );

  let hasCustomAssemblyFormat = 1;
}

def ReuseIR_MRefType : ReuseIR_Type<"MRef", "mref",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {

  let summary = "ReuseIR Mutable Reference Type";
  let description = [{
    `reuse_ir.mref` is the mutable reference that is assignable inside a unfrozen but freezable object.
  }];

  let parameters = (ins
    "mlir::Type":$pointee,
    OptionalParameter<"mlir::BoolAttr">:$atomic
  );

  let assemblyFormat = [{
    `<` $pointee (`,` `atomic` `:` $atomic^ )? `>`
  }];
}

def ReuseIR_RefType : ReuseIR_Type<"Ref", "ref",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>,DeclareTypeInterfaceMethods<ReuseIRCompositeLayoutInterface>]> {

  let summary = "ReuseIR Reference Type";
  let description = [{
    `reuse_ir.ref` is the immutable referece to the underlying type
  }];

  let parameters = (ins
    "mlir::Type":$pointee,
    OptionalParameter<"::mlir::IntegerAttr">:$rank,
    OptionalParameter<"::mlir::BoolAttr">:$strided
  );

  let assemblyFormat = [{
    `<` $pointee ( `,` `rank` `:` $rank^ `,` `strided` `:` $strided  )? `>`
  }];
}

def ReuseIR_ArrayType : ReuseIR_Type<"Array", "array",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {

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
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>,DeclareTypeInterfaceMethods<ReuseIRCompositeLayoutInterface>]> {

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
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>,DeclareTypeInterfaceMethods<ReuseIRCompositeLayoutInterface>]> {

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

def ReuseIR_CompositeType : ReuseIR_Type<"Composite", "composite",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>, DeclareTypeInterfaceMethods<ReuseIRCompositeLayoutInterface>]> {

  let summary = "ReuseIR Composite Type";
  let description = [{
    `reuse_ir.composite` is a composite type of other types.
    struct {
      Types...;
    }
  }];

  let parameters = (ins
    ArrayRefParameter<"mlir::Type">:$memberTypes
  );

  let assemblyFormat = [{
    `<` $memberTypes `>`
  }];
}

def ReuseIR_UnionType : ReuseIR_Type<"Union", "union",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>, DeclareTypeInterfaceMethods<ReuseIRCompositeLayoutInterface>]> {
  let summary = "ReuseIR Union Type";
  let description = [{
    `reuse_ir.union` is a tagged union of other types
    struct {
      <...> tag;
      union {
        Types...;
      };
    }
  }];

  let parameters = (ins
    ArrayRefParameter<"::mlir::Type">:$memberTypes
  );

  let assemblyFormat = [{
    `<` $memberTypes `>`
  }];

  let extraClassDeclaration = [{
  ::mlir::IntegerType getTagType() const {
    return ::mlir::IntegerType::get(getContext(), ::llvm::Log2_64(::llvm::PowerOf2Ceil(getMemberTypes().size())));
  }
  std::pair<::llvm::TypeSize, ::llvm::Align> getDataLayout(::mlir::DataLayout layout) const {
    ::llvm::TypeSize size = ::llvm::TypeSize::getZero();
    ::llvm::Align alignment {1};
    for (auto type : getMemberTypes()) {
        size = std::max(size, ::llvm::TypeSize::getFixed(layout.getTypeSize(type)));
        alignment = std::max(alignment, ::llvm::Align(layout.getTypeABIAlignment(type)));
    }
    size = ::llvm::alignTo(size, alignment.value());
    return {size, alignment};
  }
  }];
}

def ReuseIR_ClosureType : ReuseIR_Type<"Closure", "closure",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>, DeclareTypeInterfaceMethods<ReuseIRCompositeLayoutInterface>]> {

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
    "::mlir::Type":$outputType
  );

  let assemblyFormat = [{
    `<` `(` $inputTypes `)` `->` $outputType `>`
  }];
}


#endif // REUSE_IR_TYPES
