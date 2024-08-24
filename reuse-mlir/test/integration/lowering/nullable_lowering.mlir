// RUN: %reuse-opt %s -convert-reuse-ir-to-llvm | %FileCheck %s
!rc = !reuse_ir.rc<i64, nonatomic, frozen>
module @test {
    //CHECK: llvm.func @foo(%[[arg0:[0-9a-z]+]]: !llvm.ptr) -> !llvm.ptr {
    //CHECK:     llvm.return %[[arg0]] : !llvm.ptr
    //CHECK: }
    func.func @foo(%arg0: !rc) -> !reuse_ir.nullable<!rc> {
        %0 = reuse_ir.nullable.nonnull(%arg0 : !rc) : !reuse_ir.nullable<!rc>
        return %0 : !reuse_ir.nullable<!rc>
    }
    //CHECK: llvm.func @bar() -> !llvm.ptr {
    //CHECK:     %[[val:[0-9a-z]+]] = llvm.mlir.zero : !llvm.ptr
    //CHECK:     llvm.return %[[val]] : !llvm.ptr
    //CHECK: }
    func.func @bar() -> !reuse_ir.nullable<!rc> {
        %0 = reuse_ir.nullable.null : !reuse_ir.nullable<!rc>
        return %0 : !reuse_ir.nullable<!rc>
    }
}
