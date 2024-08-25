// RUN: %not %reuse-opt %s 2>&1 | %FileCheck %s
module @test {
    // CHECK: error: 'reuse_ir.rc.release' op must have a result when applied to a nonfreezing RC pointer
    func.func @foo(%0 : !reuse_ir.rc<i64, nonatomic, nonfreezing>) {
        reuse_ir.rc.release (%0 : !reuse_ir.rc<i64, nonatomic, nonfreezing>)
        return
    }
}
