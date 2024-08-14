#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "ReuseIR/IR/ReuseIRDialect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::reuse_ir::ReuseIRDialect>();
  mlir::registerAllExtensions(registry);
  mlir::registerAllPasses();
  return failed(mlir::MlirOptMain(
      argc, argv, "ReuseIR analysis and optimization driver\n", registry));
}
