#ifndef REUSE_IR_OPS
#define REUSE_IR_OPS

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/CommonAttrConstraints.td"

include "ReuseIR/IR/ReuseIRDialect.td"
include "ReuseIR/IR/ReuseIRAttrs.td"
include "ReuseIR/IR/ReuseIRTypes.td"

class ReuseIR_Op<string mnemonic, list<Trait> traits = []> :
    Op<ReuseIR_Dialect, mnemonic, traits>;

def AllocOp : ReuseIR_Op<"alloc"> {

  let summary = "Allocate memory token";
  let description = [{
    `reuse_ir.alloc` allocates a memory token.
    ```mlir
    reuse_ir.alloc : !reuse_ir.token<size: 16, alignment: 8>
    ```
  }];

  let results = (outs ReuseIR_TokenType:$token);

  let assemblyFormat = [{
    `:` type($token) attr-dict
  }];

  let hasVerifier = 0;
}

def IncOp : ReuseIR_Op<"inc"> {

  let summary = "Increase reference count";
  let description = [{
    `reuse_ir.inc` increases the reference count of the object.
    ```mlir
    reuse_ir.inc (%0 : !reuse_ir.rc<i32>)
    ```
  }];

  let arguments = (ins ReuseIR_RcType:$rcPtr, OptionalAttr<I64Attr>:$count);

  let assemblyFormat = [{
    `(` $rcPtr `:` type($rcPtr)  (`,` $count^ )? `)` attr-dict
  }];

  let hasVerifier = 1;
}

#endif // REUSE_IR_OPS
