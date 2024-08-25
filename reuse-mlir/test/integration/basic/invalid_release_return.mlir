// RUN: %not %reuse-opt %s 2>&1 | %FileCheck %s
module @test {
    // CHECK: error: 'reuse_ir.rc.release' op cannot have any result when applied to a frozen RC pointer
    func.func @foo(%0 : !reuse_ir.rc<i64, nonatomic, frozen>) {
        %1 = reuse_ir.rc.release (%0 : !reuse_ir.rc<i64, nonatomic, frozen>) : !reuse_ir.nullable<!reuse_ir.token<size: 8, alignment: 8>>
        return
    }
}
