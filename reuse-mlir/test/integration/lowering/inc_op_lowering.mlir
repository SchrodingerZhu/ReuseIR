// RUN: %reuse-opt %s -convert-reuse-ir-to-llvm | %FileCheck %s
module @test {
func.func @foo(%arg0: !reuse_ir.rc<i64>) {
    // CHECK: %[[REG0:[a-z0-9]+]] = llvm.getelementptr %{{[a-z0-9]+}}[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i64)>
    // CHECK: %[[REG1:[a-z0-9]+]] = llvm.load %[[REG0]] : !llvm.ptr -> i64
    // CHECK: %[[REG2:[a-z0-9]+]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK: %[[REG3:[a-z0-9]+]] = llvm.add %[[REG1]], %[[REG2]] : i64
    // CHECK: llvm.store %[[REG3]], %[[REG0]] : i64, !llvm.ptr
    reuse_ir.inc(%arg0 : <i64>, 1)
    return
}
func.func @bar(%arg0: !reuse_ir.rc<i64, atomic: true, frozen: true>) {
    // CHECK: %[[REG0:[a-z0-9]+]] = llvm.getelementptr %{{[a-z0-9]+}}[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, ptr, i64)>
    // CHECK: %[[REG1:[a-z0-9]+]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK: llvm.atomicrmw sub %[[REG0]], %[[REG1]] seq_cst : !llvm.ptr, i64
    reuse_ir.inc(%arg0 : <i64, atomic: true, frozen: true>, 1)
    return
}
}
