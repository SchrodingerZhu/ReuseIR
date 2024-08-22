// RUN: %not %reuse-opt %s 2>&1 | %FileCheck %s
!closure = !reuse_ir.closure<(i32, i32)>
module @test {
    func.func @closure_test() -> !closure {
        // CHECK: error: 'reuse_ir.closure.yield' op cannot yield a value in a closure without output
        %1 = reuse_ir.closure.new {
            ^bb(%arg0: i32, %arg1: i32):
                %2 = arith.addi %arg0, %arg1 : i32
                reuse_ir.closure.yield %2 : i32
        } : !closure
        return %1 : !closure
    }
}
