#ifndef REUSE_IR_PASSES
#define REUSE_IR_PASSES

include "mlir/Pass/PassBase.td"

def ConvertReuseIRToLLVM : Pass<"convert-reuse-ir-to-llvm"> {
  let summary = "Convert Low-level ReuseIR Operations to LLVM IR";
  let description = [{
    This pass is serves as the last step in the ReuseIR pipeline. It converts
    all remaining low-level ReuseIR operations to LLVM IR.
    ```

  }];
  let constructor = "mlir::createConvertReuseIRToLLVMPass()";
  let dependentDialects = ["::mlir::reuse_ir::ReuseIRDialect", "::mlir::LLVM::LLVMDialect"];
}

#endif // REUSE_IR_PASSES
