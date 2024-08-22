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

def ValueToRefOp : ReuseIR_Op<"val2ref"> {
    let summary = "Promote a value to stack and get the reference";
    let description = [{
      `reuse_ir.val2ref` is used to promote a value to an alloca on stack.
      For composite/union/array arguments passed as values:
        - caller should either use `byval` ptr call convention: that is, spill the
          value to stack pass its ref.
        - or when the struct is small enough, the value can be directly passed to
          target function but inorder to decompose the the object, this operaton is
          still needed.
    }];
    let results = (outs Res<ReuseIR_RefType, "", [MemAlloc<AutomaticAllocationScopeResource>]>:$result);
    let arguments = (
        ins AnyType:$value
    );
    let assemblyFormat = [{
      $value attr-dict `:` type($value) `->` type($result)
    }];
    let hasVerifier = 1;
}

def BorrowOp : ReuseIR_Op<"borrow"> {
    let summary = "Borrow the managed object and get the reference";
    let description = [{
      `reuse_ir.borrow` is used to get the reference to the managed object.
    }];
    let results = (outs ReuseIR_RefType:$result);
    let arguments = (
        ins ReuseIR_RcType:$object
    );
    let assemblyFormat = [{
      $object attr-dict `:` type($object) `->` type($result)
    }];
    let hasVerifier = 1;
}

// TODO:
// def RefToValueOp : : ReuseIR_Op<"ref2val"> {
//     let summary = "Promote a value to stack and get the reference";
//     let description = [{
//       `reuse_ir.val2ref` is used to promote a value to an alloca on stack.
//       For arguments passed as values, this operation should be used at the
//       beginning of the function to allow
//     }];
// }

def ProjOp : ReuseIR_Op<"proj"> {
  let summary = "Project a field out of a value type or a referential type";
  let description = [{
    `reuse_ir.proj` gets a reference to the field of the object similar to the
    GEP operation in LLVM IR without the first index.
    The freezing attribute is inherited from the parent reference.
  }];
  let results = (outs ReuseIR_RefType:$result);
  let arguments = (
    ins Arg<ReuseIR_RefType, "reference to the object", []>:$object,
    I64Attr:$index
  );

  let assemblyFormat = [{
    $object `[` $index `]` attr-dict `:` type($object) `->` type($result)
  }];
  let hasVerifier = 1;
}

def LoadOp : ReuseIR_Op<"load"> {
  let summary = "Load a value from a reference";
  let description = [{
    `reuse_ir.load` loads the value from the reference.
    One special thing is the mref type, which will be
    loaded as a `reuse_ir.rc` type, whose freezing state
    inherits from the parent reference.
  }];
  let results = (outs AnyType:$result);
  let arguments = (
    ins Arg<ReuseIR_RefType, "reference to the object", [MemRead]>:$object
  );

  let assemblyFormat = [{
    $object attr-dict `:` type($object) `->` type($result)
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
  let arguments = (ins Arg<ReuseIR_RefType, "destroyable object", [MemRead]>:$object);

  let assemblyFormat = [{
    `(` $object `:` type($object) `)` attr-dict
  }];
  
  // TODO: add verifier
  let hasVerifier = 0;
}

def NewClosureOp : ReuseIR_Op<"new_closure"> {
  let summary = "Create a new closure";
  let description = [{
    `reuse_ir.new_closure` creates a new closure object. 
    The closure can either be a value or wrapped in a rc pointer.
    The closure shall not capture mutable references (for now).
  }];
  let results = (outs Res<AnyTypeOf<[ReuseIR_RcTypeOf<[ReuseIR_ClosureType]>, ReuseIR_ClosureType]>, 
    "Closure Object", [MemAlloc<DefaultResource>]>:$result);
  let regions = (region MinSizedRegion<1>:$body);
  let assemblyFormat = [{
    $body `:` type($result) attr-dict
  }];
  // TODO: add verifier
  let hasVerifier = 0;
}

def ClosureYieldOp : ReuseIR_Op<"closure.yield", [ReturnLike, Terminator,
    ParentOneOf<["::mlir::reuse_ir::NewClosureOp"]>]> {
  let summary = "Yield closure return value";
  let description = [{
    `closure.yield` yields the return value of the closure.
    This serves as the terminator of the closure body.
  }];

  let arguments = (ins AnyType:$value);

  let assemblyFormat = [{
    $value `:` type($value) attr-dict
  }];
  
  // TODO: add verifier
  let hasVerifier = 0;
}
#endif // REUSE_IR_OPS
