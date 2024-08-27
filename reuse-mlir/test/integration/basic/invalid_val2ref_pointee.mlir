// RUN: %not %reuse-opt %s 2>&1 | %FileCheck %s
!test = !reuse_ir.composite<{!reuse_ir.composite<{f128, i32}>, i32}>
module @test {
    // CHECK: error: 'reuse_ir.val2ref' op must return a reference whose pointee is of the same type of the input
    func.func @foo(%arg0 : !test) {
        %1 = reuse_ir.val2ref %arg0 : !test -> !reuse_ir.ref<f128, nonfreezing>
        return
    }
}
