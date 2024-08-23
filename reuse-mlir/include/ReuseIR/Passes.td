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

def ReuseIRExpandNullability : Pass<"reuse-ir-expand-nullability", "::mlir::ModuleOp"> {
  let summary = "Expand structured control flows for nullibility checking";
  let description = [{
    This pass is used to expand hidden control flows needed for nullibility checking.
  }];
  let constructor = "::mlir::reuse_ir::createReuseIRClosureOutliningPass()";
  let dependentDialects = ["::mlir::reuse_ir::ReuseIRDialect"];
}

#endif // REUSE_IR_PASSES
