#ifndef REUSE_IR_MANGLE_INTERFACE
#define REUSE_IR_MANGLE_INTERFACE

include "mlir/IR/OpBase.td"

def ReuseIRMangleInterface : TypeInterface<"ReuseIRMangleInterface"> {
  let description = [{
    Mangle the type so that we have a unique name for its associated functions.
  }];
  let cppNamespace = "::mlir::reuse_ir";
  let methods = [
    InterfaceMethod<[{
        Return the mangled name of the type.
      }],
      /*retTy=*/"void",
      /*methodName=*/"formatMangledNameTo",
      /*args=*/(ins "::llvm::raw_ostream &":$buffer)
    >
  ];
}
#endif // REUSE_IR_MANGLE_INTERFACE
