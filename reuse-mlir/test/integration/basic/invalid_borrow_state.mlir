// RUN: %not %reuse-opt %s 2>&1 | %FileCheck %s
module @test {
    // CHECK: error: the borrowed reference must have the consistent freezing state with the RC pointer
    func.func @borrow_state_invalid(%0: !reuse_ir.rc<i32, nonatomic, nonfreezing>) {
        %1 = reuse_ir.rc.borrow %0 : 
            !reuse_ir.rc<i32, nonatomic, nonfreezing>
            -> !reuse_ir.ref<i32, frozen>
        return
    }
}
