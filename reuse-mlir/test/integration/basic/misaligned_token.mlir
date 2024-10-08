// RUN: %not %reuse-opt %s 2>&1 | %FileCheck %s
module @test {
    // CHECK: error: alignment must be a power of 2
    func.func @foo(%0 : !reuse_ir.rc<i64, nonatomic, nonfreezing>, %test: !llvm.struct<()>) {
        reuse_ir.rc.acquire (%0 : !reuse_ir.rc<i64, nonatomic, nonfreezing>)
        %1 = reuse_ir.token.alloc : !reuse_ir.token<size : 128, alignment : 13>
        reuse_ir.token.free (%1 : !reuse_ir.token<size: 128, alignment: 13>)
        return
    }
}
