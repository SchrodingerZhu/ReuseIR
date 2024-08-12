cmake_minimum_required(VERSION 3.28)
project(ReuseIR-MLIR C CXX)
set(CMAKE_CXX_STANDARD 20)

include(cmake/FindLLVM.cmake)
include(cmake/FindMLIR.cmake)
message(STATUS "include dirs for llvm and mlir: ${MLIR_INCLUDE_DIRS};${LLVM_INCLUDE_DIRS}")

add_subdirectory(include)
