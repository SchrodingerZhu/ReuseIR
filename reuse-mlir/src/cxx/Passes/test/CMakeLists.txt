add_mlir_dialect_library(MLIRReuseIRTestPasses
  STATIC
    TestAnticipatedAllocation.cpp
  DEPENDS
    MLIRReuseIR
    MLIRReuseIRAnalysis
    MLIRReuseIRPassIncGen
  LINK_COMPONENTS
    Core
  LINK_LIBS PUBLIC
    MLIRReuseIR
    MLIRReuseIRAnalysis
)
