#include "ReuseIR/IR/ReuseIRDialect.h"
#include "ReuseIR/Common.h"
#include "ReuseIR/IR/ReuseIROps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace REUSE_IR_DECL_SCOPE {
void ReuseIRDialect::initialize() {
  registerTypes();
  registerAttributes();
  addOperations<
#define GET_OP_LIST
#include "ReuseIR/IR/ReuseIROps.cpp.inc"
      >();
}
::mlir::Operation *
ReuseIRDialect::materializeConstant(::mlir::OpBuilder &builder,
                                    ::mlir::Attribute value, ::mlir::Type type,
                                    ::mlir::Location loc) {
  llvm_unreachable("TODO");
}
} // namespace REUSE_IR_DECL_SCOPE
} // namespace mlir

#include "ReuseIR/IR/ReuseIROpsDialect.cpp.inc"
