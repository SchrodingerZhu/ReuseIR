#ifndef REUSE_IR_DIALECT
#define REUSE_IR_DIALECT

include "mlir/IR/OpBase.td"

def ReuseIR_Dialect : Dialect {
  let name = "reuse_ir";

  // A short one-line summary of our dialect.
  let summary = "IR for memory reuse analysis";

  let cppNamespace = "::mlir::reuse_ir";

  let hasConstantMaterializer = 1;

  let extraClassDeclaration = [{
  void registerAttributes();
  void registerTypes();
  }];

  let dependentDialects = [
    "::mlir::BuiltinDialect",
    "::mlir::func::FuncDialect",
    "::mlir::LLVM::LLVMDialect",
  ];
}

#endif // REUSE_IR_DIALECT
