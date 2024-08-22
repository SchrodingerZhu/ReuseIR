add_mlir_dialect_library(MLIRReuseIRPasses
  STATIC
    ConvertReuseIRToLLVM.cpp
    ReuseIRClosureOutlining.cpp
  DEPENDS
    MLIRReuseIR
    MLIRReuseIRPassIncGen
  LINK_COMPONENTS
    Core
  LINK_LIBS PUBLIC
    MLIRReuseIR
)
