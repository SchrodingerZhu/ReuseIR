// RUN: %reuse-opt %s 2>&1 -reuse-ir-expand-control-flow | %FileCheck %s
!tk = !reuse_ir.token<size: 64, alignment: 16>
module @test attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"} {
    // CHECK: func.func @token_ensure_expansion(%[[regarg0:[a-z0-9]+]]: !reuse_ir.nullable<!reuse_ir.token<size : 64, alignment : 16>>) -> !reuse_ir.token<size : 64, alignment : 16> {
    // CHECK:     %[[reg0:[a-z0-9]+]] = reuse_ir.nullable.check(%[[regarg0]] : <!reuse_ir.token<size : 64, alignment : 16>>) -> i1
    // CHECK:     %[[reg1:[a-z0-9]+]] = scf.if %[[reg0]] -> (!reuse_ir.token<size : 64, alignment : 16>) {
    // CHECK:     %[[reg2:[a-z0-9]+]] = reuse_ir.nullable.coerce(%[[regarg0]] : <!reuse_ir.token<size : 64, alignment : 16>>) : !reuse_ir.token<size : 64, alignment : 16>
    // CHECK:     scf.yield %[[reg2]] : !reuse_ir.token<size : 64, alignment : 16>
    // CHECK:     } else {
    // CHECK:     %[[reg2:[a-z0-9]+]] = reuse_ir.token.alloc : <size : 64, alignment : 16>
    // CHECK:     scf.yield %[[reg2]] : !reuse_ir.token<size : 64, alignment : 16>
    // CHECK:     }
    // CHECK:     return %[[reg1]] : !reuse_ir.token<size : 64, alignment : 16>
    // CHECK: }
    
    func.func @token_ensure_expansion(%0: !reuse_ir.nullable<!tk>) -> !tk{
        %1 = reuse_ir.token.ensure (%0 : !reuse_ir.nullable<!tk>) : !tk
        return %1 : !tk
    }
}
