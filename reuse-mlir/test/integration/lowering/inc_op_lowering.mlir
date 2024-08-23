// RUN: %reuse-opt %s -convert-reuse-ir-to-llvm | %FileCheck %s
module @test {
func.func @foo(%arg0: !reuse_ir.rc<i64, nonatomic, nonfreezing>) {
    // CHECK: %[[REG0:[a-z0-9]+]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK: %[[REG1:[a-z0-9]+]] = llvm.getelementptr %{{[a-z0-9]+}}[0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i64, i64)>
    // CHECK: %[[REG2:[a-z0-9]+]] = llvm.load %[[REG1]] : !llvm.ptr -> i64
    // CHECK: %[[REG3:[a-z0-9]+]] = llvm.add %[[REG2]], %[[REG0]] : i64
    // CHECK: llvm.store %[[REG3]], %[[REG1]] : i64, !llvm.ptr
    reuse_ir.rc.acquire(%arg0 : <i64, nonatomic, nonfreezing>)
    return
}
func.func @bar(%arg0: !reuse_ir.rc<i64, nonatomic, frozen>) {
    // CHECK: llvm.call @__reuse_ir_acquire_freezable(%{{[a-z0-9]+}}) : (!llvm.ptr) -> ()
    reuse_ir.rc.acquire(%arg0 : <i64, nonatomic, frozen>)
    return
}
}
