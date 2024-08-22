// RUN: %reuse-opt %s | %FileCheck %s
!closure = !reuse_ir.closure<(i32, i32) -> i32>
module @test {
    // CHECK: func.func @closure_test() -> !reuse_ir.rc<!reuse_ir.closure<(i32, i32) -> i32>, nonatomic, nonfreezing>
    func.func @closure_test() -> !reuse_ir.rc<!closure, nonatomic, nonfreezing> {
        %1 = reuse_ir.new_closure {
            ^bb(%arg0: i32, %arg1: i32):
                %2 = arith.addi %arg0, %arg1 : i32
                reuse_ir.closure.yield %2 : i32
        } : !reuse_ir.rc<!closure, nonatomic, nonfreezing>
        return %1 : !reuse_ir.rc<!closure, nonatomic, nonfreezing>
    }
}
