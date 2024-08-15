configure_file(lit.site.cfg.py.in lit.site.cfg.py @ONLY)

find_file(filecheck_path FileCheck HINTS ${LLVM_TOOLS_BINARY_DIR} REQUIRED)
message(STATUS "FileCheck path: ${filecheck_path}")

find_file(not_path not HINTS ${LLVM_TOOLS_BINARY_DIR} REQUIRED)
message(STATUS "not path: ${not_path}")

add_custom_target(check
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/lit "${CMAKE_CURRENT_BINARY_DIR}" -v
  DEPENDS reuse-opt)
