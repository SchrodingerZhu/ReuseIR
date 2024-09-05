// RUN: %reuse-opt %s -reuse-ir-token-reuse | %FileCheck %s
!rc64 = !reuse_ir.rc<i64, nonatomic, nonfreezing>
!rc64x2 = !reuse_ir.rc<!reuse_ir.composite<"test" {i64, i64}>, nonatomic, nonfreezing>
module @test attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"} {
// CHECK: func.func @reuse(%[[regarg0:[a-z0-9]+]]: !reuse_ir.rc<i64, nonatomic, nonfreezing>) {
// CHECK:  %[[reg0:[a-z0-9]+]] = reuse_ir.rc.release(%[[regarg0]] : <i64, nonatomic, nonfreezing>) : !reuse_ir.nullable<!reuse_ir.token<size : 16, alignment : 8>>
// CHECK:  reuse_ir.token.free(%[[reg0]] : !reuse_ir.nullable<!reuse_ir.token<size : 16, alignment : 8>>)
// CHECK:  return
// CHECK: }
    func.func @reuse(%0: !rc64) {
        %1 = reuse_ir.rc.release (%0 : !rc64) : !reuse_ir.nullable<!reuse_ir.token<size: 16, alignment: 8>>
        return
    }
}
