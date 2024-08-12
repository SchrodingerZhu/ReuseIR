#ifndef REUSE_IR_TYPES
#define REUSE_IR_TYPES

include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/IR/BuiltinTypes.td"
include "ReuseIR/IR/ReuseIRDialect.td"

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

def ReuseIR_BoxType : ReuseIR_Type<"Box", "box",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {
  let summary = "ReuseIR Control Box";
  let description = [{
    `reuse_ir.box` is the underlying control box of RC managed objects
  }];
  let parameters = (ins
    "mlir::Type":$pointee
  );
  let assemblyFormat = [{
    `<` $pointee `>`
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
    OptionalParameter<"mlir::BoolAttr">:$atomic
  );

  let assemblyFormat = [{
    `<` $pointee ( `,` `atomic` `:` $atomic^ )? `>`
  }];
}

def ReuseIR_RefType : ReuseIR_Type<"Ref", "ref",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {

  let summary = "ReuseIR Reference Type";
  let description = [{
    `reuse_ir.ref` is the immutable referece to the underlying type
  }];

  let parameters = (ins
    "mlir::Type":$pointee,
    OptionalParameter<"::mlir::IndexAttr">:$rank,
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
    "::llvm::ArrayRef<::mlir::IndexAttr>":$sizes
  );

  let assemblyFormat = [{
    `<` $elementType `,` $sizes `>`
  }];
}

def ReuseIR_VectorType : ReuseIR_Type<"Vector", "vector",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {

  let summary = "ReuseIR Vector Type";
  let description = [{
    `reuse_ir.vector` is a single-dimension vector with dynamic length
  }];

  let parameters = (ins
    "mlir::Type":$elementType
  );

  let assemblyFormat = [{
    `<` $elementType `>`
  }];
}

def ReuseIR_OpaqueType : ReuseIR_Type<"Opaque", "opaque",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {

  let summary = "ReuseIR Opaque Type";
  let description = [{
    `reuse_ir.opaque` is an opaque type for externally managed object.
  }];

  let parameters = (ins
    "mlir::IndexAttr":$size,
    "mlir::IndexAttr":$alignment
  );

  let assemblyFormat = [{
    `<` $size `,` $alignment `>`
  }];
}

def ReuseIR_CompositeType : ReuseIR_Type<"Composite", "composite",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {

  let summary = "ReuseIR Composite Type";
  let description = [{
    `reuse_ir.composite` is a composite type of other types
  }];

  let parameters = (ins
    "::llvm::ArrayRef<::mlir::Type>":$memberTypes,
    OptionalParameter<"::llvm::StringRef">:$name
  );

  let assemblyFormat = [{
    `<` $memberTypes ( `,` `name` `:` $name^ )? `>`
  }];
}

def ReuseIR_UnitType : ReuseIR_Type<"Unit", "unit",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {

  let summary = "ReuseIR Unit Type";
  let description = [{
    `reuse_ir.unit` is the unit type.
  }];
}

def ReuseIR_UnionType : ReuseIR_Type<"Union", "union",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {

  let summary = "ReuseIR Union Type";
  let description = [{
    `reuse_ir.union` is a tagged union of other types
  }];

  let parameters = (ins
    "::llvm::ArrayRef<::mlir::Type>":$memberTypes,
    OptionalParameter<"::llvm::StringRef">:$name
  );

  let assemblyFormat = [{
    `<` $memberTypes ( `,` `name` `:` $name^ )? `>`
  }];
}

def ReuseIR_ClosureType : ReuseIR_Type<"Closure", "closure",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {

  let summary = "ReuseIR Closure Type";
  let description = [{
    `reuse_ir.closure` is for capturing lambda expressions 
  }];

  let parameters = (ins
    "::llvm::ArrayRef<::mlir::Type>":$inputTypes,
    "::mlir::Type":$outputType
  );

  let assemblyFormat = [{
    `<` `(` $inputTypes `)` `->` $outputType `>`
  }];
}


#endif // REUSE_IR_TYPES