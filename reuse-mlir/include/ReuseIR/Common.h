#pragma once
#include "llvm/ADT/StringRef.h"
#include <algorithm>
#include <array>
#if defined(__GNUG__) && __has_attribute(visibility)
#define REUSE_IR_DECL_SCOPE [[gnu::visibility("hidden")]] reuse_ir
#else
#define REUSE_IR_DECL_SCOPE reuse_ir
#endif

#if __has_include("mlir/Support/LogicalResult.h")
#include "mlir/Support/LogicalResult.h"
namespace mlir::reuse_ir {
using mlir::LogicalResult;
inline LogicalResult success() { return LogicalResult::success(); }
} // namespace mlir::reuse_ir
#else
#include "llvm/Support/LogicalResult.h"
namespace mlir::reuse_ir {
using llvm::LogicalResult;
inline LogicalResult success() { return LogicalResult::success(); }
} // namespace mlir::reuse_ir
#endif

namespace mlir::reuse_ir {
template <size_t N> struct StringLiteral {
  constexpr StringLiteral(const char (&str)[N]) {
    std::copy_n(str, N, value.begin());
  }
  std::array<char, N> value;
  constexpr operator llvm::StringRef() const { return {&value[0], N - 1}; }
};

template <size_t N> StringLiteral(const char (&str)[N]) -> StringLiteral<N>;

template <StringLiteral Str> constexpr inline decltype(Str) operator""_str() {
  return Str;
}

} // namespace mlir::reuse_ir
