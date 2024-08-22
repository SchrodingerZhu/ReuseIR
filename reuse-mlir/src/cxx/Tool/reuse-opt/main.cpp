#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "ReuseIR/IR/ReuseIRDialect.h"
#include "ReuseIR/Passes.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::reuse_ir::ReuseIRDialect>();
  mlir::registerAllExtensions(registry);
  mlir::registerAllPasses();
  mlir::reuse_ir::registerReuseIRPasses();
  return failed(mlir::MlirOptMain(
      argc, argv, "ReuseIR analysis and optimization driver\n", registry));
}
