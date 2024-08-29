add_mlir_dialect_library(MLIRReuseIRAnalysis
  STATIC
    AliasAnalysis.cpp
    AnticipatedAllocation.cpp
  DEPENDS
    MLIRReuseIR
  LINK_COMPONENTS
    Core
  LINK_LIBS PUBLIC
    MLIRReuseIR
)
