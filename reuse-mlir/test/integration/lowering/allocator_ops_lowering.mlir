// RUN: %reuse-opt %s -convert-reuse-ir-to-llvm | %FileCheck %s
module @test {
func.func @foo() -> !reuse_ir.token<size : 512, alignment : 16> {
    // CHECK: %[[REG0:[a-z0-9]+]] = llvm.mlir.constant(512 : i64) : i64
    // CHECK: %[[REG1:[a-z0-9]+]] = llvm.mlir.constant(16 : i64) : i64
    // CHECK: %[[REG2:[a-z0-9]+]] = llvm.call @__reuse_ir_alloc(%[[REG0]], %[[REG1]]) : (i64, i64) -> !llvm.ptr
    %token = reuse_ir.token.alloc : !reuse_ir.token<size : 512, alignment : 16>
    // CHECK: llvm.return %[[REG2]]
    return %token : !reuse_ir.token<size : 512, alignment : 16>
}
func.func @baz(%token: !reuse_ir.token<size : 8, alignment : 8>) {
    // CHECK: %[[REG0:[a-z0-9]+]] = llvm.mlir.constant(8 : i64) : i64
    // CHECK: %[[REG1:[a-z0-9]+]] = llvm.mlir.constant(8 : i64) : i64
    // CHECK: llvm.call @__reuse_ir_dealloc(%{{[a-z0-9]+}}, %[[REG0]], %[[REG1]]) : (!llvm.ptr, i64, i64) -> ()
    reuse_ir.token.free (%token : !reuse_ir.token<size : 8, alignment : 8>)
    return
}
}
