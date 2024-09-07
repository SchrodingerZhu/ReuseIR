add_mlir_dialect(ReuseIROps reuse_ir)

set(LLVM_TARGET_DEFINITIONS ReuseIROps.td)
mlir_tablegen(ReuseIROpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(ReuseIROpsEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(ReuseIROpsStructs.h.inc -gen-attrdef-decls)
mlir_tablegen(ReuseIROpsStructs.cpp.inc -gen-attrdef-defs)
mlir_tablegen(ReuseIROpsAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(ReuseIROpsAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRReuseIREnumsIncGen)
add_dependencies(mlir-headers MLIRReuseIREnumsIncGen)

add_reuseir_mlir_doc(ReuseIROps ReuseIROps ./ --gen-op-doc)
# FIXME: uncomment the following line once we have added an attribute.
# add_reuseir_mlir_doc(ReuseIRAttrs ReuseIRAttrs ./ --gen-attrdef-doc)
add_reuseir_mlir_doc(ReuseIRTypes ReuseIRTypes ./ --gen-typedef-doc)
add_reuseir_mlir_doc(ReuseIRDialect ReuseIRDialect ./ --gen-dialect-doc)
