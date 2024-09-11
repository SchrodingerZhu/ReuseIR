#include "ReuseIR/IR/ReuseIRDialect.h"
#include "mlir/CAPI/Registration.h"
namespace mlir {
namespace reuse_ir {
extern "C" {
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(ReuseIR, reuse_ir, ReuseIRDialect)
}
} // namespace reuse_ir
} // namespace mlir
