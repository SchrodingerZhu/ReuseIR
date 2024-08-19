// RUN: %reuse-opt %s | %FileCheck %s
module @test {
    // CHECK: func.func @foo(%{{[a-z0-9]+}}: !reuse_ir.rc<i64>, %{{[a-z0-9]+}}: !llvm.struct<()>) {
    func.func @foo(%0 : !reuse_ir.rc<i64>, %test: !llvm.struct<()>) {
        reuse_ir.inc (%0 : !reuse_ir.rc<i64>, 1)
        %1 = reuse_ir.alloc : !reuse_ir.token<size : 128, alignment : 16>
        reuse_ir.free (%1 : !reuse_ir.token<size: 128, alignment: 16>)
        return
    }
    // CHECK: func.func @bar(%{{[a-z0-9]+}}: !reuse_ir.rc<i64, atomic : true>) {
    func.func @bar(%0 : !reuse_ir.rc<i64, atomic : true>) {
        return
    }
    // CHECK: func.func @baz(%{{[a-z0-9]+}}: !reuse_ir.rc<i64, frozen : true>) {
    func.func @baz(%0 : !reuse_ir.rc<i64, frozen : true>) {
        return
    }
    // CHECK: func.func @qux(%arg0: !reuse_ir.rc<i64, frozen : false, atomic : true>) {
    func.func @qux(%0 : !reuse_ir.rc<i64, frozen : false, atomic : true>) {
        return
    }
    func.func @projection(%0: !reuse_ir.rc<!reuse_ir.composite<!reuse_ir.composite<i32, i32>, i32>>) {
        %1 = reuse_ir.borrow %0 : 
            !reuse_ir.rc<!reuse_ir.composite<!reuse_ir.composite<i32, i32>, i32>>
            -> !reuse_ir.ref<!reuse_ir.composite<!reuse_ir.composite<i32, i32>, i32>>
        %2 = reuse_ir.proj %1[0, 1] : 
            !reuse_ir.ref<!reuse_ir.composite<!reuse_ir.composite<i32, i32>, i32>> -> i32
        return
    }
    func.func @projection_ref(%0: !reuse_ir.rc<!reuse_ir.composite<!reuse_ir.composite<i32, i32>, i32>>) {
        %1 = reuse_ir.borrow %0 : 
            !reuse_ir.rc<!reuse_ir.composite<!reuse_ir.composite<i32, i32>, i32>>
            -> !reuse_ir.ref<!reuse_ir.composite<!reuse_ir.composite<i32, i32>, i32>>
        %2 = reuse_ir.proj as_reference %1[0, 1] : 
            !reuse_ir.ref<!reuse_ir.composite<!reuse_ir.composite<i32, i32>, i32>> -> !reuse_ir.ref<!reuse_ir.composite<i32, i32>>
        return
    }
}
