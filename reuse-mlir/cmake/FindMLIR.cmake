find_package(MLIR REQUIRED CONFIG)

message(STATUS "Found MLIR ${MLIR_PACKAGE_VERSION}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

include(${MLIR_DIR}/AddMLIR.cmake)

include_directories(${MLIR_INCLUDE_DIRS})
separate_arguments(MLIR_DEFINITIONS_LIST NATIVE_COMMAND ${MLIR_DEFINITIONS})
add_definitions(${MLIR_DEFINITIONS_LIST})
