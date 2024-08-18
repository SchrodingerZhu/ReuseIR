get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
add_mlir_dialect_library(MLIRReuseIR
  STATIC
    ReuseIRDialect.cpp
    ReuseIRTypes.cpp
    ReuseIRAttrs.cpp
    ReuseIROps.cpp
  DEPENDS
    MLIRReuseIRInterfaces
    MLIRReuseIROpsIncGen
  LINK_COMPONENTS
    Core
  LINK_LIBS PUBLIC
    ${dialect_libs}
    MLIRReuseIRInterfaces
)
