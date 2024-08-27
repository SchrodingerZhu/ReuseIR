#ifndef REUSE_IR_DIALECT
#define REUSE_IR_DIALECT

include "mlir/IR/OpBase.td"

def ReuseIR_Dialect : Dialect {
  let name = "reuse_ir";

  // A short one-line summary of our dialect.
  let summary = "IR for memory reuse analysis";

  let cppNamespace = "::mlir::reuse_ir";

  let hasConstantMaterializer = 1;
  let useDefaultTypePrinterParser = 0;

  let extraClassDeclaration = [{
  void registerAttributes();
  void registerTypes();
  
  Type parseType(DialectAsmParser &parser) const override final;
  void printType(Type type, DialectAsmPrinter &printer) const override final;
  }];

  let dependentDialects = [
    "::mlir::BuiltinDialect",
    "::mlir::func::FuncDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::LLVM::LLVMDialect",
  ];
}

#endif // REUSE_IR_DIALECT
