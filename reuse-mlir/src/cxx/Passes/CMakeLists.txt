add_mlir_dialect_library(MLIRReuseIRPasses
  STATIC
    ConvertReuseIRToLLVM.cpp
    ClosureOutlining.cpp
    ExpandControlFlow.cpp
    AcquireReleaseFusion.cpp
    InferUnionTag.cpp
    PrintReuseAnalysis.cpp
    TokenReuse.cpp
    GenFreezableVTable.cpp
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
