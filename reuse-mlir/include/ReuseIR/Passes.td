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

def ReuseIRGenFreezableVTable : Pass<"reuse-ir-gen-freezable-vtable", "::mlir::ModuleOp"> {
  let summary = "Generate vtables for freezable rc objects";
  let description = [{
    This pass generates vtables for freezable rc objects.
  }];
  let constructor = "::mlir::reuse_ir::createReuseIRGenFreezableVTablePass()";
  let dependentDialects = ["::mlir::reuse_ir::ReuseIRDialect", "::mlir::scf::SCFDialect"];
}

def ReuseIRExpandControlFlow : Pass<"reuse-ir-expand-control-flow", "::mlir::ModuleOp"> {
  let summary = "Expand structured control flows for ReuseIR operations";
  let description = [{
    This pass is used to expand control flows for ReuseIR operations.
    - reuse_ir.token.ensure
    - reuse_ir.token.free (with nullible token)
    - reuse_ir.rc.release
  }];
    let options = [
    Option<"outlineNestedRelease", "outline-nested-release", "bool", "false", "Outline nested RC release operations">
  ];
  let constructor = "::mlir::reuse_ir::createReuseIRExpandControlFlowPass()";
  let dependentDialects = ["::mlir::reuse_ir::ReuseIRDialect", "::mlir::scf::SCFDialect"];
}

def ReuseIRAcquireReleaseFusion : Pass<"reuse-ir-acquire-release-fusion", "::mlir::func::FuncOp"> {
  let summary = "Fuse ReuseIR acquire and release Operations";
  let description = [{
    This pass is used to fuse acquire and release operations.
  }];
  let constructor = "::mlir::reuse_ir::createReuseIRAcquireReleaseFusionPass()";
  let dependentDialects = ["::mlir::reuse_ir::ReuseIRDialect"];
}

def ReuseIRInferUnionTag : Pass<"reuse-ir-infer-union-tag", "::mlir::func::FuncOp"> {
  let summary = "Infer union tags for ReuseIR operations";
  let description = [{
    This pass is used to infer union tags for ReuseIR operations.
  }];
  let constructor = "::mlir::reuse_ir::createReuseIRInferUnionTagPass()";
  let dependentDialects = ["::mlir::reuse_ir::ReuseIRDialect"];
}

def ReuseIRPrintReuseAnalysis : Pass<"reuse-ir-print-reuse-analysis", "::mlir::ModuleOp"> {
  let summary = "Print Reuse Analysis";
  let description = [{
    This pass is used to print the reuse analysis information.
  }];
  let constructor = "::mlir::reuse_ir::createReuseIRPrintReuseAnalysisPass()";
  let dependentDialects = ["::mlir::reuse_ir::ReuseIRDialect"];
}

def ReuseIRTokenReuse : Pass<"reuse-ir-token-reuse", "::mlir::ModuleOp"> {
  let summary = "Apply reuse analysis to reuse token and recyle unused tokens";
  let description = [{
    This pass is used to apply reuse analysis to reuse token and recyle unused tokens.
  }];
  let constructor = "::mlir::reuse_ir::createReuseIRTokenReusePass()";
  let dependentDialects = ["::mlir::reuse_ir::ReuseIRDialect"];
}

#endif // REUSE_IR_PASSES
