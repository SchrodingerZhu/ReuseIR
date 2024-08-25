// RUN: %reuse-opt %s 2>&1 -convert-reuse-ir-to-llvm | %FileCheck %s
!rc = !reuse_ir.rc<i64, nonatomic, frozen>
!arc = !reuse_ir.rc<i64, atomic, frozen>
module @test attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"} {
    func.func @rc_release(%0: !rc) {
        // CHECK: llvm.call @__reuse_ir_release_freezable(%{{[a-z0-9]+}}) : (!llvm.ptr) -> ()
        reuse_ir.rc.release (%0 : !rc)
        return
    }

    func.func @rc_release_atomic(%0: !arc) {
        // CHECK: llvm.call @__reuse_ir_release_atomic_freezable(%{{[a-z0-9]+}}) : (!llvm.ptr) -> ()
        reuse_ir.rc.release (%0 : !arc)
        return
    }
}
