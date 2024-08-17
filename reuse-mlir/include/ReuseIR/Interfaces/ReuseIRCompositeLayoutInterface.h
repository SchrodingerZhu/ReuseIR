#pragma once

#include "ReuseIR/Common.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <variant>

namespace mlir {
namespace reuse_ir {
class CompositeLayout {
public:
  using FieldKind = std::variant<mlir::Type, size_t>;
  struct Field {
    size_t index;
    llvm::Align alignment;
  };

private:
  llvm::Align alignment = llvm::Align{1};
  llvm::TypeSize size = llvm::TypeSize::getZero();
  llvm::SmallVector<FieldKind> raw_fields;
  llvm::DenseMap<size_t, Field> field_map;

public:
  const llvm::Align &getAlignment() const { return alignment; }
  const llvm::TypeSize &getSize() const { return size; }
  llvm::ArrayRef<FieldKind> getRawFields() const { return raw_fields; }
  Field getField(size_t idx) const { return field_map.at(idx); }

  CompositeLayout(mlir::DataLayout layout, llvm::ArrayRef<mlir::Type> fields);

  mlir::LLVM::LLVMStructType
  getLLVMType(mlir::LLVMTypeConverter &converter) const;
};
} // namespace reuse_ir
} // namespace mlir

#include "ReuseIR/Interfaces/ReuseIRCompositeLayoutInterface.h.inc"

namespace mlir {
namespace reuse_ir {
class CompositeLayoutCache {
  mlir::DataLayout dataLayout;
  llvm::DenseMap<ReuseIRCompositeLayoutInterface, CompositeLayout> cache;

public:
  CompositeLayoutCache(mlir::DataLayout dataLayout)
      : dataLayout(dataLayout), cache{} {}
  const CompositeLayout &get(const ReuseIRCompositeLayoutInterface &iface);
  mlir::DataLayout getDataLayout() const { return dataLayout; }
};
} // namespace reuse_ir
} // namespace mlir
