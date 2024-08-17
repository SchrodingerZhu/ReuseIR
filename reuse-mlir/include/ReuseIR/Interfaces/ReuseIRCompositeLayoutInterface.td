#ifndef REUSE_IR_COMPOSITE_LAYOUT_INTERFACE
#define REUSE_IR_COMPOSITE_LAYOUT_INTERFACE

include "mlir/IR/OpBase.td"

def ReuseIRCompositeLayoutInterface : TypeInterface<"ReuseIRCompositeLayoutInterface"> {
  let description = [{
    Unified interface for converting type to LLVM struct layout.
  }];
  let cppNamespace = "::mlir::reuse_ir";
  let methods = [
    InterfaceMethod<[{
        Get the CompositeLayout of the type.
      }],
      /*retTy=*/"::mlir::reuse_ir::CompositeLayout",
      /*methodName=*/"getCompositeLayout",
      /*args=*/(ins "::mlir::DataLayout":$dataLayout)
    >
  ];
}
#endif // REUSE_IR_COMPOSITE_LAYOUT_INTERFACE
