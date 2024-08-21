// RUN: %not %reuse-opt %s 2>&1 | %FileCheck %s
module @test {
    func.func @load(%0: !reuse_ir.ref<f128, nonfreezing>) -> f32 {
        // CHECK: error: 'reuse_ir.load' op expected to return a value of 'f128', but 'f32' is found instead
        %1 = reuse_ir.load %0 : !reuse_ir.ref<f128, nonfreezing> -> f32
        return %1 : f32
    }
}
