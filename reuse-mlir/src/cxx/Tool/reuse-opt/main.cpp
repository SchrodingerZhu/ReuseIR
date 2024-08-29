#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "ReuseIR/IR/ReuseIRDialect.h"
#include "ReuseIR/Passes.h"

#ifdef REUSE_IR_ENABLE_TESTS
namespace mlir::reuse_ir {
std::unique_ptr<Pass> createTestAnticipatedAllocationPass();
} // namespace mlir::reuse_ir
#endif

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::reuse_ir::ReuseIRDialect>();
  mlir::registerAllExtensions(registry);
  mlir::registerAllPasses();
  mlir::reuse_ir::registerReuseIRPasses();
#ifdef REUSE_IR_ENABLE_TESTS
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::reuse_ir::createTestAnticipatedAllocationPass();
  });
#endif
  return failed(mlir::MlirOptMain(
      argc, argv, "ReuseIR analysis and optimization driver\n", registry));
}
