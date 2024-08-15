// RUN: %not %reuse-opt %s 2>&1 | %FileCheck %s
module @test {
    // CHECK: error: Alignment must be a power of 2
    func.func @foo(%0 : !reuse_ir.rc<i64>, %test: !llvm.struct<()>) {
        reuse_ir.inc (%0 : !reuse_ir.rc<i64>, 1)
        %1 = reuse_ir.alloc : !reuse_ir.token<size : 128, alignment : 13>
        reuse_ir.free (%1 : !reuse_ir.token<size: 128, alignment: 13>)
        return
    }
}