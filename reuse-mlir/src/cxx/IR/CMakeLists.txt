add_mlir_dialect_library(ReuseIR
    ReuseIROps.cpp
  LINK_COMPONENTS
    Core
  LINK_LIBS
    cxx-rs
)