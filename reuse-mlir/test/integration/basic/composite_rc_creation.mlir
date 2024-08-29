// RUN: %reuse-opt %s 
!inner = !reuse_ir.composite<{i32, i32, f128}>
!inner_rc = !reuse_ir.rc<!inner, nonatomic, nonfreezing>
!outer = !reuse_ir.composite<{!inner_rc, i32}>
!outer_rc = !reuse_ir.rc<!outer, nonatomic, nonfreezing>

module @test {
    func.func @projection(%a: i32, %b: i32, %c: f128, %d: i32) -> !outer_rc {
        %1 = reuse_ir.composite.assemble (%a, %b, %c) : (i32, i32, f128) -> !inner
        %2 = reuse_ir.rc.create value(%1) : (!inner) -> !inner_rc
        %3 = reuse_ir.composite.assemble (%2, %d) : (!inner_rc, i32) -> !outer
        %4 = reuse_ir.rc.create value(%3) : (!outer) -> !outer_rc
        return %4 : !outer_rc
    }
}
