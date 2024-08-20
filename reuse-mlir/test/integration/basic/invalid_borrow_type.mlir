// RUN: %not %reuse-opt %s 2>&1 | %FileCheck %s
module @test {
    // CHECK: error: the borrowed reference must have the consistent pointee type with the RC pointer
    func.func @borrow_state_invalid(%0: !reuse_ir.rc<i32, nonatomic, frozen>) {
        %1 = reuse_ir.borrow %0 : 
            !reuse_ir.rc<i32, nonatomic, frozen>
            -> !reuse_ir.ref<i64, frozen>
        return
    }
}