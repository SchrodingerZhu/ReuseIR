#ifndef REUSE_IR_PASSES
#define REUSE_IR_PASSES

include "mlir/Pass/PassBase.td"

def ConvertReuseIRToLLVM : Pass<"convert-reuse-ir-to-llvm", "::mlir::ModuleOp"> {
  let summary = "Convert Low-level ReuseIR Operations to LLVM IR";
  let description = [{
    This pass is serves as the last step in the ReuseIR pipeline. It converts
    all remaining low-level ReuseIR operations to LLVM IR.
    ```

  }];
  let constructor = "::mlir::reuse_ir::createConvertReuseIRToLLVMPass()";
  let dependentDialects = ["::mlir::reuse_ir::ReuseIRDialect", "::mlir::LLVM::LLVMDialect"];
}

def ReuseIRClosureOutlining : Pass<"reuse-ir-closure-outlining", "::mlir::ModuleOp"> {
  let summary = "Outlines ReuseIR Closures";
  let description = [{
    This pass outlines closures and instantiates their virtual tables.
  }];
  let constructor = "::mlir::reuse_ir::createReuseIRClosureOutliningPass()";
  let dependentDialects = ["::mlir::reuse_ir::ReuseIRDialect"];
}

def ReuseIRExpandControlFlow : Pass<"reuse-ir-expand-control-flow", "::mlir::ModuleOp"> {
  let summary = "Expand structured control flows for ReuseIR operations";
  let description = [{
    This pass is used to expand control flows for ReuseIR operations.
    - reuse_ir.token.ensure
    - reuse_ir.token.free (with nullible token)
    - reuse_ir.rc.release
  }];
  let constructor = "::mlir::reuse_ir::createReuseIRExpandControlFlowPass()";
  let dependentDialects = ["::mlir::reuse_ir::ReuseIRDialect", "::mlir::scf::SCFDialect"];
}

#endif // REUSE_IR_PASSES
