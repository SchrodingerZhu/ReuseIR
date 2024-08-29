if (REUSE_IR_ENABLE_TESTS)
  add_subdirectory(test)
endif()
add_mlir_dialect_library(MLIRReuseIRPasses
  STATIC
    ConvertReuseIRToLLVM.cpp
    ClosureOutlining.cpp
    ExpandControlFlow.cpp
    AcquireReleaseFusion.cpp
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
