get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
add_mlir_dialect_library(MLIRReuseIR
  STATIC
    ReuseIRDialect.cpp
    ReuseIRTypes.cpp
    ReuseIRAttrs.cpp
    ReuseIROps.cpp
    ReuseIROpsEnums.cpp
  DEPENDS
    MLIRReuseIRInterfaces
    MLIRReuseIROpsIncGen
    MLIRReuseIREnumsIncGen
  LINK_COMPONENTS
    Core
  LINK_LIBS PUBLIC
    ${dialect_libs}
    MLIRReuseIRInterfaces
)
