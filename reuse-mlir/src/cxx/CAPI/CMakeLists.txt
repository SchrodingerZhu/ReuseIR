add_mlir_dialect_library(MLIRReuseIRCAPI
  STATIC
    CAPI.cpp
  DEPENDS
    MLIRReuseIR
    MLIRReuseIRAnalysis
    MLIRReuseIRPasses
    MLIRReuseIRInterfaces
  LINK_COMPONENTS
    Core
  LINK_LIBS PUBLIC
    MLIRReuseIR
    MLIRReuseIRAnalysis
    MLIRReuseIRPasses
    MLIRReuseIRInterfaces
)
