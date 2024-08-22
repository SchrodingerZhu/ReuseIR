// RUN: %not %reuse-opt %s 2>&1 | %FileCheck %s
!closure = !reuse_ir.closure<(i32, i32) -> i32>
module @test {
    func.func @closure_test() -> !closure {
        // CHECK: error: 'reuse_ir.closure.new' op the number of arguments in the region must match the number of input types in the closure type
        %1 = reuse_ir.closure.new {
            ^bb(%arg0: i32):
                reuse_ir.closure.yield %arg0 : i32
        } : !closure
        return %1 : !closure
    }
}
