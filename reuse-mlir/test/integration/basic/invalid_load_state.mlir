// RUN: %not %reuse-opt %s 2>&1 | %FileCheck %s
module @test {
    func.func @load(%0: !reuse_ir.ref<!reuse_ir.mref<f128, nonatomic>, nonfreezing>) -> !reuse_ir.rc<f128, nonatomic, nonfreezing> {
        // CHECK: error: 'reuse_ir.load' op cannot load a mutable RC pointer through a nonfreezing reference
        %1 = reuse_ir.load %0 : !reuse_ir.ref<!reuse_ir.mref<f128, nonatomic>, nonfreezing> -> !reuse_ir.rc<f128, nonatomic, nonfreezing>
        return %1 : !reuse_ir.rc<f128, nonatomic, nonfreezing>
    }
}
