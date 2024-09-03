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
include "mlir/IR/BuiltinAttributes.td"

include "ReuseIR/IR/ReuseIRDialect.td"
include "ReuseIR/IR/ReuseIRAttrs.td"
include "ReuseIR/IR/ReuseIRTypes.td"

class ReuseIR_Op<string mnemonic, list<Trait> traits = []> :
    Op<ReuseIR_Dialect, mnemonic, traits>;

def TokenAllocOp : ReuseIR_Op<"token.alloc", []> {

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
}

def TokenFreeOp : ReuseIR_Op<"token.free", []> {
  let summary = "Free memory token";
  let description = [{
    `reuse_ir.token.free` deallocates a memory token.
    ```mlir
    reuse_ir.token.free (%x : !reuse_ir.token<size: 16, alignment: 8>)
    ```
  }];

  let arguments = (ins Res<
    AnyTypeOf<[ReuseIR_TokenType, ReuseIR_NullableToken]>,
    "Free Memory Token", [MemFree<DefaultResource>]>:$token);

  let assemblyFormat = [{
    `(` $token `:` type($token) `)` attr-dict
  }];
}

def TokenReinterpretOp : ReuseIR_Op<"token.reinterpret"> {
  let summary = "Reinterpret the token type";
  let description = [{
    `reuse_ir.token.reinterpret` reinterprets the token type.
  }];
  let results = (outs Res<ReuseIR_RefType, "reinterpreted reference">:$result);
  let arguments = (ins Arg<ReuseIR_TokenType, "Token to reinterpret">:$token);
  let assemblyFormat = [{
    `(` $token `:` type($token) `)` attr-dict `:` type($result)
  }];
}

def TokenEnsureOp : ReuseIR_Op<"token.ensure"> {
  let summary = "Ensure a nonnull token";
  let description = [{
    `reuse_ir.token.ensure` converts a possibly nullable token to a token by checking
    the nullability and allocate memory on need.
  }];
  let results = (outs Res<ReuseIR_TokenType, "output token", [MemAlloc<DefaultResource>]>:$result);
  let arguments = (ins Arg<ReuseIR_NullableToken, "input token">:$nullableToken);
  let assemblyFormat = [{
    `(` $nullableToken `:` type($nullableToken) `)` attr-dict `:` type($result)
  }];
}

def TokenReallocOp : ReuseIR_Op<"token.realloc"> {
  let summary = "Ensure a nonnull token";
  let description = [{
    `reuse_ir.token.realloc` converts a possibly nullable token to a token by checking
    the nullability and reallocate the memory on need. Utimately, we should custom the
    allocator to avoid memcpy if the content is not to be retained.
  }];
  let results = (outs Res<ReuseIR_TokenType, "output token", [MemAlloc<DefaultResource>]>:$result);
  let arguments = (ins Arg<ReuseIR_NullableToken, "input token">:$nullableToken);
  let assemblyFormat = [{
    `(` $nullableToken `:` type($nullableToken) `)` attr-dict `->` type($result)
  }];
}

def NullableCheckOp : ReuseIR_Op<"nullable.check"> {
  let summary = "Check the nullibility of a pointer";
  let description = [{
    `reuse_ir.nullable.check` returns 1 if target pointer is null or 0 otherwise.
  }];
  let results = (outs Res<I1, "result">:$result);
  let arguments = (ins Arg<ReuseIR_NullableType, "input">:$nullable);
  let assemblyFormat = [{
    `(` $nullable `:` type($nullable) `)` attr-dict `->` type($result)
  }];
}

def NullableCoerceOp : ReuseIR_Op<"nullable.coerce"> {
  let summary = "Force unwrap the nullable as a nonnull pointer";
  let description = [{
    `reuse_ir.nullable.coerce` forcibly unwrap the nullable as a nonnull pointer.
  }];
  let results = (outs Res<AnyTypeOf<[ReuseIR_RcType, ReuseIR_TokenType]>, "result">:$result);
  let arguments = (ins Arg<ReuseIR_NullableType, "input token">:$nullable);
  let assemblyFormat = [{
    `(` $nullable `:` type($nullable) `)` attr-dict `:` type($result)
  }];
}

def NullableNullOp : ReuseIR_Op<"nullable.null"> {
  let summary = "Create a null pointer";
  let description = [{
    `reuse_ir.nullable.null` creates a null pointer.
  }];
  let results = (outs Res<ReuseIR_NullableType, "result">:$result);
  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
}

def NullableNonNullOp : ReuseIR_Op<"nullable.nonnull"> {
  let summary = "Create a nullable pointer from a nonnull value";
  let description = [{
    `reuse_ir.nullable.nonnull` creates a nullable pointer from a nonnull value.
  }];
  let results = (outs Res<ReuseIR_NullableType, "result">:$result);
  let arguments = (ins Arg<AnyTypeOf<[ReuseIR_RcType, ReuseIR_TokenType]>, "pointer">:$pointer);
  let assemblyFormat = [{
    `(` $pointer `:` type($pointer) `)` attr-dict `:` type($result)
  }];
}

def RcAcquireOp : ReuseIR_Op<"rc.acquire"> {

  let summary = "Increase reference count";
  let description = [{
    `reuse_ir.rc.acquire` increases the reference count of the object.
    ```mlir
    reuse_ir.rc.acquire (%0 : !reuse_ir.rc<i32>)
    ```
  }];

  let arguments = (ins Arg<ReuseIR_RcType, "target rc pointer", [MemReadAt<0>, MemReadAt<1>]>:$rcPtr);

  let assemblyFormat = [{
    `(` $rcPtr `:` type($rcPtr) `)` attr-dict
  }];

  let hasVerifier = 1;
}

def RcDecreaseOp : ReuseIR_Op<"rc.decrease"> {
  let summary = "Decrease reference count";
  let description = [{
    This decrement operation is for progressively lowering rc.release operation. It decreases
    the reference count and return a flag indicating whether the object should be dropped.
  }];

  let arguments = (ins Arg<ReuseIR_RcType, "target rc pointer", [MemReadAt<0>, MemWriteAt<1>]>:$rcPtr);
  let results = (outs Res<I1, "drop flag">:$dropFlag);
  let assemblyFormat = [{
    `(` $rcPtr `:` type($rcPtr) `)` attr-dict `:`  type($dropFlag) 
  }];

  let hasVerifier = 1;
}

def RcReleaseOp : ReuseIR_Op<"rc.release"> {
  let summary = "Release reference count and drop the object if necessary";
  let description = [{
    `reuse_ir.rc.release` decreases the reference count of the object.
    - It target rc is a freezable one, it has no result. The operation translates to runtime function call.
    - If the target rc is not freezable, it returns a nullable token.
  }];

  let arguments = (ins 
    Arg<ReuseIR_RcType, "target rc pointer", [MemReadAt<0>, MemWriteAt<1>]>:$rcPtr,
    OptionalAttr<IndexAttr>:$tag
  );
  let results = (outs Res<Optional<ReuseIR_NullableToken>, "token">:$token);
  let assemblyFormat = [{
    `(` $rcPtr `:` type($rcPtr) `)` (`tag` `(` $tag^ `)`)? attr-dict  (`:` type($token)^)?
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

def RcBorrowOp : ReuseIR_Op<"rc.borrow"> {
    let summary = "Borrow the managed object and get the reference";
    let description = [{
      `reuse_ir.rc.borrow` is used to get the reference to the managed object.
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

def RcTokenizeOp : ReuseIR_Op<"rc.tokenize"> {
    let summary = "Convert the rc to a token that is suitable for its underlying RcBox";
    let description = [{
      `reuse_ir.rc.tokenize` converts the rc to a token that is suitable for its underlying RcBox.
    }];
    let results = (outs ReuseIR_TokenType:$token);
    let arguments = (
        ins ReuseIR_RcType:$object
    );
    let assemblyFormat = [{
      $object attr-dict `:` type($object) `->` type($token)
    }];
}

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
    IndexAttr:$index
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
      - For RC type, it applies the `reuse_ir.rc.release` operation.
      - For nested composite/union types, it recursively applies the 
        `reuse_ir.destroy` operation.
    - Specialized types such as Opaque/Vector/Closure are specially handled.
  }];

  // TODO: Add type restrictions
  let arguments = (ins 
    Arg<ReuseIR_RefType, "destroyable object", [MemRead, MemFree<DefaultResource>]>:$object,
    OptionalAttr<IndexAttr>:$tag
  );

  let assemblyFormat = [{
    `(` $object `:` type($object) `)` (`tag` `(` $tag^ `)`)? attr-dict
  }];
  
  // TODO: add verifier
  let hasVerifier = 0;
}

def UnionGetTagOp : ReuseIR_Op<"union.get_tag"> {
  let summary = "Get the tag of the union type";
  let description = [{
    `reuse_ir.union.get_tag` gets the tag of the union type.
  }];
  let results = (outs Builtin_Index:$tag);
  let arguments = (ins ReuseIR_RefTypeOf<[ReuseIR_UnionType]>:$unionRef);
  let assemblyFormat = [{
    $unionRef attr-dict `:` type($unionRef) `->` type($tag)
  }];
}

def UnionInspectOp : ReuseIR_Op<"union.inspect"> {
  let summary = "Inspect the union type";
  let description = [{
    `reuse_ir.union.inspect` inspects the union type and returns a reference to the type of
    the given index. Such inspection is not checked in runtime.
  }];
  let results = (outs Optional<ReuseIR_RefType>:$result);
  let arguments = (ins ReuseIR_RefTypeOf<[ReuseIR_UnionType]>:$unionRef, IndexAttr:$index);
  let assemblyFormat = [{
    $unionRef `[` $index `]` attr-dict `:` type($unionRef) (`->` type($result)^)?
  }];
}

def PanicOp : ReuseIR_Op<"panic"> {
  let summary = "Panic the program";
  let description = [{
    `reuse_ir.panic` panics the program.
  }];
  let arguments = (ins Builtin_StringAttr:$message);
  let results = (outs Optional<AnyType>:$result);
  let assemblyFormat = [{
    `(` $message `)` attr-dict (`:` type($result)^)?
  }];
}

def UnreachableOp : ReuseIR_Op<"unreachable"> {
  let summary = "Notify the program that the code path is unreachable";
  let description = [{
    `reuse_ir.unreachable` notifies the program that the code path is unreachable.
  }];
  let results = (outs Optional<AnyType>:$result);
  let assemblyFormat = [{
    attr-dict (`:` type($result)^)?
  }];
}

def ClosureNewOp : ReuseIR_Op<"closure.new"> {
  let summary = "Create a new closure";
  let description = [{
    `reuse_ir.closure.new` creates a new closure object (without wrapping it into RC). 
    The closure shall not capture mutable references (for now).
  }];
  let results = (outs Res<ReuseIR_ClosureType, 
    "Closure Object", [MemAlloc<DefaultResource>]>:$closure);
  let regions = (region VariadicRegion<MinSizedRegion<1>>:$body);
  let assemblyFormat = [{
    ($body^)? `:` type($closure) attr-dict
  }];
  let extraClassDeclaration = [{
    ::mlir::reuse_ir::ClosureType getClosureType();
  }];
  // TODO: add verifier
  let hasVerifier = 1;
}

def ClosureAssembleOp : ReuseIR_Op<"closure.assemble"> {
  let summary = "Create a new closure with given resources";
  let description = [{
    `reuse_ir.closure.assemble` creates a new closure object. 
    The closure can either be a value or wrapped in a rc pointer.
    The closure shall not capture mutable references (for now).
  }];
  let results = (outs Res<ReuseIR_ClosureType, 
    "Closure Object", [MemAlloc<DefaultResource>]>:$closure);
  let arguments = (ins FlatSymbolRefAttr:$vtable, 
    Arg<Optional<ReuseIR_TokenType>, "memory resource for argpack", [MemWrite]>:$argpack);
  let assemblyFormat = [{
    `vtable` `(` $vtable `)`
    (`argpack` `(` $argpack^ `:` type($argpack) `)` )?
    `:` attr-dict type($closure)
  }];
}

def ClosureYieldOp : ReuseIR_Op<"closure.yield", [ReturnLike, Terminator,
    ParentOneOf<["::mlir::reuse_ir::ClosureNewOp"]>]> {
  let summary = "Yield closure return value";
  let description = [{
    `closure.yield` yields the return value of the closure.
    This serves as the terminator of the closure body.
  }];

  let arguments = (ins Optional<AnyType>:$value);

  let assemblyFormat = [{
   ($value^ `:` type($value))? attr-dict
  }];
  
  // TODO: add verifier
  let hasVerifier = 1;
}

def ClosureVTableOp : ReuseIR_Op<"closure.vtable"> {
  let summary = "Define a closure vtable";
  let description = [{
    `closure.vtable` defines a vtable for the closure.
    The vtable is a list of function pointers.
  }];

  let arguments = (ins FlatSymbolRefAttr:$name, TypeAttr:$closureType, FlatSymbolRefAttr:$func, FlatSymbolRefAttr:$clone, FlatSymbolRefAttr:$drop);

  let assemblyFormat = [{
    attr-dict $name `{` 
      `closure_type` `:` $closureType `,`
      `func` `:` $func `,`
      `clone` `:` $clone `,`
      `drop` `:` $drop
    `}` 
  }];
}

def CompositeAssembleOp : ReuseIR_Op<"composite.assemble"> {
  let summary = "Create a new composite object";
  let description = [{
    `reuse_ir.composite.assemble` creates a new composite object.
  }];
  let results = (outs Res<ReuseIR_CompositeType, "Composite Object">:$composite);
  let arguments = (ins Variadic<AnyType>:$fields);
  let assemblyFormat = [{
    `(` $fields `)` attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def UnionAssembleOp : ReuseIR_Op<"union.assemble"> {
  let summary = "Create a new union object";
  let description = [{
    `reuse_ir.union.assemble` creates a new union object.
  }];
  let results = (outs Res<ReuseIR_UnionType, "Union Object">:$result);
  let arguments = (ins Arg<IndexAttr, "tag">:$tag, AnyType:$field);
  let assemblyFormat = [{
    `(` $tag `,` $field `)` attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def RcCreateOp : ReuseIR_Op<"rc.create", [AttrSizedOperandSegments]> {
  let summary = "Create a new Rc object";
  let description = [{
    `reuse_ir.rc.create` creates a new Rc object.
  }];
  let results = (outs Res<ReuseIR_RcType, "Rc Object", [MemAlloc<DefaultResource>]>:$rc);
  let arguments = (ins 
    AnyType:$value, 
    Optional<ReuseIR_NullableToken>:$token,
    Optional<ReuseIR_RegionCtxType>:$region
  );
  let assemblyFormat = [{
    `value` `(` $value `)`
    (`token` `(` $token^ `)` )?
    (`region` `(` $region^ `)`)?
    attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

#endif // REUSE_IR_OPS
