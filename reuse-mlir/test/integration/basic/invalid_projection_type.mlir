// RUN: %not %reuse-opt %s 2>&1 | %FileCheck %s
!test = !reuse_ir.composite<!reuse_ir.composite<i32, i32, f128>, i32>
module @test {
    func.func @projection(%0: !reuse_ir.rc<!test, nonatomic, nonfreezing>) {
        %1 = reuse_ir.rc.borrow %0 : 
            !reuse_ir.rc<!test, nonatomic, nonfreezing>
            -> !reuse_ir.ref<!test, nonfreezing>
        // CHECK: error: 'reuse_ir.proj' op expected to return a reference to '!reuse_ir.composite<i32, i32, f128>', but found a reference to '!reuse_ir.composite<i32, i32>'
        %2 = reuse_ir.proj %1[0] : 
            !reuse_ir.ref<!test, nonfreezing> -> !reuse_ir.ref<!reuse_ir.composite<i32, i32>, nonfreezing>
        return
    }
}
