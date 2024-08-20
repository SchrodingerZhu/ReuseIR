// RUN: %not %reuse-opt %s 2>&1 | %FileCheck %s
module @test {
    // CHECK: error: the amount of increment must be non-zero
    func.func @foo(%0 : !reuse_ir.rc<i64, nonatomic, nonfreezing>) {
        reuse_ir.inc (%0 : !reuse_ir.rc<i64, nonatomic, nonfreezing>, 0)
        return
    }
}
