// RUN: %reuse-opt %s 
!inner = !reuse_ir.composite<{i32, i32, f128}>
!inner_rc = !reuse_ir.rc<!inner, nonatomic, nonfreezing>
!outer = !reuse_ir.composite<{!inner_rc, i32}>
!outer_rc = !reuse_ir.rc<!outer, nonatomic, nonfreezing>

module @test attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"} {
    func.func @projection(%a: i32, %b: i32, %c: f128, %d: i32) -> !outer_rc {
        %1 = reuse_ir.composite.assemble (%a, %b, %c) : (i32, i32, f128) -> !inner
        %tk0 = reuse_ir.token.alloc : !reuse_ir.token<size: 48, alignment: 16>
        %2 = reuse_ir.rc.create value(%1) token(%tk0) : (!inner, !reuse_ir.token<size: 48, alignment: 16>) -> !inner_rc
        %tk1 = reuse_ir.token.alloc : !reuse_ir.token<size: 24, alignment: 8>
        %3 = reuse_ir.composite.assemble (%2, %d) : (!inner_rc, i32) -> !outer
        %4 = reuse_ir.rc.create value(%3) token(%tk1) : (!outer, !reuse_ir.token<size: 24, alignment: 8>) -> !outer_rc
        return %4 : !outer_rc
    }
}
