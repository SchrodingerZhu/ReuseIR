get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
add_mlir_dialect_library(ReuseIR
  SHARED
    ReuseIRDialect.cpp
    ReuseIRTypes.cpp
    ReuseIRAttrs.cpp
  DEPENDS
    MLIRReuseIROpsIncGen
  LINK_COMPONENTS
    Core
  LINK_LIBS PUBLIC
    cxx-rs
    ${dialect_libs}
)
