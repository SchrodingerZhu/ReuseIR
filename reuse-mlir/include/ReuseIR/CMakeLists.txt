add_subdirectory(IR)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name ReuseIR)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix ReuseIR)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix ReuseIR)
add_public_tablegen_target(MLIRReuseIRPassIncGen)
