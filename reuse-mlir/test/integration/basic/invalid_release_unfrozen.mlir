// RUN: %not %reuse-opt %s 2>&1 | %FileCheck %s
module @test {
    // CHECK: error: 'reuse_ir.rc.release' op cannot be applied to an unfrozen RC pointer
    func.func @foo(%0 : !reuse_ir.rc<i64, nonatomic, unfrozen>) {
        reuse_ir.rc.release (%0 : !reuse_ir.rc<i64, nonatomic, unfrozen>)
        return
    }
}
