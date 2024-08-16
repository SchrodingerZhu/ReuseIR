add_mlir_dialect_library(MLIRReuseIRPasses
  SHARED
    ConvertReuseIRToLLVM.cpp
  DEPENDS
    MLIRReuseIR
    MLIRReuseIRPassIncGen
  LINK_COMPONENTS
    Core
  LINK_LIBS PUBLIC
    MLIRReuseIR
)
