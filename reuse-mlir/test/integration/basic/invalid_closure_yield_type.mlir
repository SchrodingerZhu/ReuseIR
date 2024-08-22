// RUN: %not %reuse-opt %s 2>&1 | %FileCheck %s
!closure = !reuse_ir.closure<(i32, i32) -> i32>
module @test {
    func.func @closure_test() -> !reuse_ir.rc<!closure, nonatomic, nonfreezing> {
        // CHECK: error: 'reuse_ir.closure.yield' op expected to yield a value of 'i32', but 'index' is found instead
        %1 = reuse_ir.closure.new {
            ^bb(%arg0: i32, %arg1: i32):
                %2 = arith.addi %arg0, %arg1 : i32
                %3 = arith.index_castui %2 : i32 to index
                reuse_ir.closure.yield %3 : index
        } : !reuse_ir.rc<!closure, nonatomic, nonfreezing>
        return %1 : !reuse_ir.rc<!closure, nonatomic, nonfreezing>
    }
}
