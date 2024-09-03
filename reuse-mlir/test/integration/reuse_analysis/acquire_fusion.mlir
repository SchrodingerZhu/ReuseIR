// RUN: %reuse-opt %s -reuse-ir-expand-control-flow=outline-nested-release=0 -reuse-ir-acquire-release-fusion 2>&1 | %FileCheck %s
!rc = !reuse_ir.rc<i64, nonatomic, nonfreezing>
!refrc = !reuse_ir.ref<!rc, nonfreezing>
!struct = !reuse_ir.composite<{!rc, !rc, !rc}>
!box = !reuse_ir.rc<!struct, nonatomic, nonfreezing>
!ref = !reuse_ir.ref<!struct, nonfreezing>
!tk1 = !reuse_ir.token<size: 32, alignment: 8>
!tk2 = !reuse_ir.token<size: 16, alignment: 8>
module @test attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"} {
    // CHECK: reuse_ir.rc.acquire(%{{[a-z0-9]+}} : <i64, nonatomic, nonfreezing>)
    // CHECK-NEXT: reuse_ir.rc.acquire(%{{[a-z0-9]+}} : <i64, nonatomic, nonfreezing>)
    // CHECK-NEXT: %[[REG:[a-z0-9]+]] = reuse_ir.nullable.null : <!reuse_ir.token<size : 32, alignment : 8>>
    // CHECK-NEXT: scf.yield %[[REG]] : !reuse_ir.nullable<!reuse_ir.token<size : 32, alignment : 8>>
    func.func @fusion(%0: !box) -> !rc {
        %ref = reuse_ir.rc.borrow %0 : !box -> !ref
        %proj = reuse_ir.proj %ref [1] : !ref -> !refrc
        %valrc = reuse_ir.load %proj : !refrc -> !rc
        %proj2 = reuse_ir.proj %ref [2] : !ref -> !refrc
        %valrc2 = reuse_ir.load %proj2 : !refrc -> !rc
        reuse_ir.rc.acquire (%valrc2 : !rc)
        reuse_ir.rc.acquire (%valrc : !rc)
        %tk = reuse_ir.rc.release (%0 : !box) : !reuse_ir.nullable<!tk1>
        %tk2 = reuse_ir.rc.release (%valrc2 : !rc) : !reuse_ir.nullable<!tk2>
        reuse_ir.token.free (%tk : !reuse_ir.nullable<!tk1>)
        reuse_ir.token.free (%tk2 : !reuse_ir.nullable<!tk2>)
        return %valrc : !rc
    }
}
