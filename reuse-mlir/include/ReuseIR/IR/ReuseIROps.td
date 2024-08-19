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

def AllocOp : ReuseIR_Op<"alloc", []> {

  let summary = "Allocate memory token";
  let description = [{
    `reuse_ir.alloc` allocates a memory token.
    ```mlir
    reuse_ir.alloc : !reuse_ir.token<size: 16, alignment: 8>
    ```
  }];

  let results = (outs Res<
    ReuseIR_TokenType,
    "Allocated Memory Token", [MemAlloc<DefaultResource>]>:$token);

  let assemblyFormat = [{
    `:` type($token) attr-dict
  }];

  let hasVerifier = 0;
}

def FreeOp : ReuseIR_Op<"free", []> {
  let summary = "Free memory token";
  let description = [{
    `reuse_ir.free` deallocates a memory token.
    ```mlir
    reuse_ir.free (%x : !reuse_ir.token<size: 16, alignment: 8>)
    ```
  }];

  let arguments = (ins Res<
    ReuseIR_TokenType,
    "Free Memory Token", [MemFree<DefaultResource>]>:$token);

  let assemblyFormat = [{
    `(` $token `:` type($token) `)` attr-dict
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

  let arguments = (ins Arg<ReuseIR_RcType, "RC pointer", [MemReadAt<0>, MemReadAt<1>]>:$rcPtr, OptionalAttr<I64Attr>:$count);

  let assemblyFormat = [{
    `(` $rcPtr `:` type($rcPtr)  (`,` $count^ )? `)` attr-dict
  }];

  let hasVerifier = 1;
}

def ProjectableType : AnyTypeOf<[ReuseIR_RcType, ReuseIR_RefType, ReuseIR_CompositeType, ReuseIR_ArrayType]>;

def ProjOp : ReuseIR_Op<"proj"> {
  let summary = "Project a field out of a value type or a referential type";
  let description = [{
    `reuse_ir.proj` projects a field out of a value type or a referential type.
    The reference counter is not mantained in this operation.
    ```mlir
    %0 = reuse_ir.proj %0[0] : !reuse_ir.rc<!reuse_ir.composite<i32, i32>> -> i32
    ```
    For value type, this translates to llvm.extract_value. 
    For referential type, this translates to a GEP chain with/without a load depending on the asRef attribute.
    If there is no index, the projection loads the whole object behind a referential type.
    ```mlir
    %0 = reuse_ir.proj %0 : !reuse_ir.rc<i32> -> i32
    ```
    Multiple indices can be used to access nested value-based composite/array types.
    ```mlir
    %1 = reuse_ir.proj %0[0, 1] : 
            !reuse_ir.rc<!reuse_ir.composite<!reuse_ir.composite<i32, i32>, i32>> -> i32
    ```
    It is not allowed to project into nested referential types. 
    In such cases, one should first project the outer referential type and then project the 
    inner referential type.
    For `reuse.mref` inside a freezable RC object, the projected result is a RC pointer with the
    same frozen attribute as the original object.
  }];
  let results = (outs AnyType:$result);
  let arguments = (
    ins Arg<ProjectableType, "projectable object", [MemRead]>:$object,
    OptionalAttr<I64ArrayAttr>:$indices,
    OptionalAttr<UnitAttr>:$asReference
  );

  let assemblyFormat = [{
    (custom<AsReferenceAttr>($asReference)^)? $object ($indices^)? attr-dict `:` type($object) `->` type($result)
  }];

  let hasVerifier = 1;
}

def DestroyOp : ReuseIR_Op<"destroy"> {
  let summary = "Destroy a value type or a referential type";
  let description = [{
    - For composite/union type (either a reference or the value), 
      this operation project out the fields that require additional 
      maintenance and apply the required operations.
      - For RC type, it applies the `reuse_ir.dec` operation.
      - For nested composite/union types, it recursively applies the 
        `reuse_ir.destroy` operation.
    - Specialized types such as Opaque/Vector/Closure are specially handled.
  }];

  // TODO: Add type restrictions
  let arguments = (ins Arg<AnyType, "destroyable object", [MemRead]>:$object);

  let assemblyFormat = [{
    `(` $object `:` type($object) `)` attr-dict
  }];
  
  // TODO: add verifier
  let hasVerifier = 0;
}

#endif // REUSE_IR_OPS
