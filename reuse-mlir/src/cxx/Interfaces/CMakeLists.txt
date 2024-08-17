get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
add_mlir_dialect_library(MLIRReuseIRInterfaces
  SHARED
    ReuseIRMangleInterface.cpp
  DEPENDS
    MLIRReuseIRMangleInterfaceIncGen
  LINK_COMPONENTS
    Core
  LINK_LIBS PUBLIC
    ${dialect_libs}
    MLIRIR
    MLIRSupport
)
