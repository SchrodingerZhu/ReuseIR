// RUN: %reuse-opt %s 2>&1
!incomplete = !reuse_ir.composite<"test" incomplete>
!test = !reuse_ir.composite<"test" {i32, !reuse_ir.mref<!incomplete, nonatomic>}>

module @test {
    func.func @foo(%0: !reuse_ir.rc<!test, nonatomic, frozen>) {
        %1 = reuse_ir.rc.borrow %0 : 
            !reuse_ir.rc<!test, nonatomic, frozen>
            -> !reuse_ir.ref<!test, frozen>
        %2 = reuse_ir.proj %1[1] : 
            !reuse_ir.ref<!test, frozen> -> !reuse_ir.ref<!reuse_ir.mref<!test, nonatomic>, frozen>
        %3 = reuse_ir.load %2 : !reuse_ir.ref<!reuse_ir.mref<!test, nonatomic>, frozen> -> !reuse_ir.nullable<!reuse_ir.rc<!test, nonatomic, frozen>>
        return
    }
}
