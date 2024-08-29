// RUN: %reuse-opt %s -test-anticipated-allocation | %FileCheck %s
!inner = !reuse_ir.composite<{i32, i32, f128}>
!inner_rc = !reuse_ir.rc<!inner, nonatomic, nonfreezing>
!outer = !reuse_ir.composite<{!inner_rc, i32}>
!outer_rc = !reuse_ir.rc<!outer, nonatomic, nonfreezing>

module @test {
    // CHECK:      func.func @test(%[[regarg0:[a-z0-9]+]]: i32, %[[regarg1:[a-z0-9]+]]: i32, %[[regarg2:[a-z0-9]+]]: f128, %[[regarg3:[a-z0-9]+]]: i32) -> !reuse_ir.rc<!reuse_ir.composite<{!reuse_ir.rc<!reuse_ir.composite<{i32, i32, f128}>, nonatomic, nonfreezing>, i32}>, nonatomic, nonfreezing> {
    // CHECK-NEXT:  %[[reg0:[a-z0-9]+]] = reuse_ir.composite.assemble(%arg0, %arg1, %arg2) {anticipated_allocation = [!reuse_ir.token<size : {{24|48}}, alignment : {{8|16}}>, !reuse_ir.token<size : {{24|48}}, alignment : {{8|16}}>]} : (i32, i32, f128) -> !reuse_ir.composite<{i32, i32, f128}>
    // CHECK-NEXT:  %[[reg1:[a-z0-9]+]] = reuse_ir.rc.create value(%[[reg0]]) {anticipated_allocation = [!reuse_ir.token<size : {{24|48}}, alignment : {{8|16}}>, !reuse_ir.token<size : {{24|48}}, alignment : {{8|16}}>]} : (!reuse_ir.composite<{i32, i32, f128}>) -> !reuse_ir.rc<!reuse_ir.composite<{i32, i32, f128}>, nonatomic, nonfreezing>
    // CHECK-NEXT:  %[[reg2:[a-z0-9]+]] = reuse_ir.composite.assemble(%[[reg1]], %[[regarg3]]) {anticipated_allocation = [!reuse_ir.token<size : 24, alignment : 8>]} : (!reuse_ir.rc<!reuse_ir.composite<{i32, i32, f128}>, nonatomic, nonfreezing>, i32) -> !reuse_ir.composite<{!reuse_ir.rc<!reuse_ir.composite<{i32, i32, f128}>, nonatomic, nonfreezing>, i32}>
    // CHECK-NEXT:  %[[reg3:[a-z0-9]+]] = reuse_ir.rc.create value(%[[reg2]]) {anticipated_allocation = [!reuse_ir.token<size : 24, alignment : 8>]} : (!reuse_ir.composite<{!reuse_ir.rc<!reuse_ir.composite<{i32, i32, f128}>, nonatomic, nonfreezing>, i32}>) -> !reuse_ir.rc<!reuse_ir.composite<{!reuse_ir.rc<!reuse_ir.composite<{i32, i32, f128}>, nonatomic, nonfreezing>, i32}>, nonatomic, nonfreezing>
    // CHECK-NEXT:  return {anticipated_allocation = []} %[[reg3]] : !reuse_ir.rc<!reuse_ir.composite<{!reuse_ir.rc<!reuse_ir.composite<{i32, i32, f128}>, nonatomic, nonfreezing>, i32}>, nonatomic, nonfreezing>
    // CHECK-NEXT: }
    func.func @test(%a: i32, %b: i32, %c: f128, %d: i32) -> !outer_rc {
        %1 = reuse_ir.composite.assemble (%a, %b, %c) : (i32, i32, f128) -> !inner
        %2 = reuse_ir.rc.create value(%1) : (!inner) -> !inner_rc
        %3 = reuse_ir.composite.assemble (%2, %d) : (!inner_rc, i32) -> !outer
        %4 = reuse_ir.rc.create value(%3) : (!outer) -> !outer_rc
        return %4 : !outer_rc
    }
}
