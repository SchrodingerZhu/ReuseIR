// RUN: %reuse-opt %s | %FileCheck %s
!test = !reuse_ir.composite<!reuse_ir.composite<i32, i32>, i32>
module @test {
    
    // CHECK: func.func @foo(%{{[0-9a-z]+}}: !reuse_ir.rc<i64, atomic, nonfreezing>, %{{[0-9a-z]+}}: !llvm.struct<()>)
    func.func @foo(%0 : !reuse_ir.rc<i64, atomic, nonfreezing>, %test: !llvm.struct<()>) {
        reuse_ir.inc (%0 : !reuse_ir.rc<i64, atomic, nonfreezing>, 1)
        %1 = reuse_ir.alloc : !reuse_ir.token<size : 128, alignment : 16>
        reuse_ir.free (%1 : !reuse_ir.token<size: 128, alignment: 16>)
        return
    }
    // CHECK: func.func @bar(%{{[0-9a-z]+}}: !reuse_ir.rc<i64, atomic, nonfreezing>)
    func.func @bar(%0 : !reuse_ir.rc<i64, atomic, nonfreezing>) {
        return
    }
    // CHECK: func.func @baz(%{{[0-9a-z]+}}: !reuse_ir.rc<i64, nonatomic, frozen>)
    func.func @baz(%0 : !reuse_ir.rc<i64, nonatomic, frozen>) {
        return
    }
    // CHECK: func.func @qux(%{{[0-9a-z]+}}: !reuse_ir.rc<i64, atomic, unfrozen>)
    func.func @qux(%0 : !reuse_ir.rc<i64, atomic, unfrozen>) {
        return
    }
    func.func @projection(%0: !reuse_ir.rc<!test, nonatomic, nonfreezing>) {
        %1 = reuse_ir.borrow %0 : 
            !reuse_ir.rc<!test, nonatomic, nonfreezing>
            -> !reuse_ir.ref<!test, nonfreezing>
        %2 = reuse_ir.proj %1[0] : 
            !reuse_ir.ref<!test, nonfreezing> -> !reuse_ir.ref<!reuse_ir.composite<i32, i32>, nonfreezing>
        %3 = reuse_ir.proj %2[1] : 
            !reuse_ir.ref<!reuse_ir.composite<i32, i32>, nonfreezing> -> !reuse_ir.ref<i32, nonfreezing> 
        return
    }
}
