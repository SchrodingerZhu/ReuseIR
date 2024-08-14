#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#pragma GCC visibility push(hidden)
#define GET_OP_CLASSES
#include "ReuseIR/IR/ReuseIROpsDialect.h.inc"
#pragma GCC visibility pop
