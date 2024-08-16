// RUN: %not %reuse-opt %s 2>&1 | %FileCheck %s
module @test {
    // CHECK: error: size must be a multiple of alignment
    func.func @foo() {
        %1 = reuse_ir.alloc : !reuse_ir.token<size : 129, alignment : 16>
        reuse_ir.free (%1 : !reuse_ir.token<size: 129, alignment: 16>)
        return
    }
}
