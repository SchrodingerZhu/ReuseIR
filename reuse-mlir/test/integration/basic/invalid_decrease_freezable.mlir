// RUN: %not %reuse-opt %s 2>&1 | %FileCheck %s
module @test {
    // CHECK: error: 'reuse_ir.rc.decrease' op can only be applied to a nonfreezing RC pointer
    func.func @foo(%0 : !reuse_ir.rc<i64, nonatomic, frozen>) {
        %1 = reuse_ir.rc.decrease (%0 : !reuse_ir.rc<i64, nonatomic, frozen>) : i1
        return
    }
}
