// RUN: %not %reuse-opt %s 2>&1 | %FileCheck %s
module @test {
    // CHECK: error: 'reuse_ir.inc' op cannot increase a non-frozen but freezable RC pointer
    func.func @foo(%0 : !reuse_ir.rc<i64, nonatomic, unfrozen>) {
        reuse_ir.inc (%0 : !reuse_ir.rc<i64, nonatomic, unfrozen>)
        return
    }
}
