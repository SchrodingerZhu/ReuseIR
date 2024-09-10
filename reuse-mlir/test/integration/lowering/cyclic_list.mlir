// RUN: %reuse-opt %s \
// RUN:   -reuse-ir-gen-freezable-vtable \
// RUN:   -reuse-ir-expand-control-flow=outline-nested-release=1 \
// RUN:   -convert-scf-to-cf \
// RUN:   -canonicalize \
// RUN:   -convert-reuse-ir-to-llvm \
// RUN:   -convert-to-llvm -reconcile-unrealized-casts | %mlir-translate -mlir-to-llvmir | %opt -O3 -S | %FileCheck %s
!ilist = !reuse_ir.composite<"list" incomplete>
!mref = !reuse_ir.mref<!ilist, nonatomic>
!list = !reuse_ir.composite<"list" {i64, !mref}>
!urclist = !reuse_ir.rc<!list, nonatomic, unfrozen>
!frclist = !reuse_ir.rc<!list, nonatomic, frozen>
!token = !reuse_ir.token<size: 40, alignment: 8>
!nullable = !reuse_ir.nullable<!urclist>
!fref = !reuse_ir.ref<!ilist, frozen>
!uref = !reuse_ir.ref<!ilist, unfrozen>
!muref = !reuse_ir.ref<!mref, unfrozen>
!nullabletk = !reuse_ir.nullable<!token>
module @test attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"} {
    func.func private @opaque(%0 : !fref) 
    func.func @test()  {
        %1 = reuse_ir.region.run {
          ^bb(%ctx : !reuse_ir.region_ctx):
          %null = reuse_ir.nullable.null : !nullable
          %0 = arith.constant 0 : i64
          %head = reuse_ir.composite.assemble (%0, %null) : (i64, !nullable) -> !list
          %head_tk = reuse_ir.token.alloc : !token
          %rchead = reuse_ir.rc.create value(%head) token(%head_tk) region(%ctx) : (!list, !token, !reuse_ir.region_ctx) -> !urclist
          %lb = arith.constant 1 : index
          %ub = arith.constant 10 : index
          %res = scf.for %i = %lb to %ub step %lb iter_args(%acc = %rchead) -> (!urclist) {
            %i_ = arith.index_cast %i : index to i64
            %next = reuse_ir.composite.assemble (%i_, %null) : (i64, !nullable) -> !list
            %next_tk = reuse_ir.token.alloc : !token
            %rcnext = reuse_ir.rc.create value(%next) token(%next_tk) region(%ctx) : (!list, !token, !reuse_ir.region_ctx) -> !urclist
            %acc_uref = reuse_ir.rc.borrow %acc : !urclist -> !uref
            %acc_muref = reuse_ir.proj %acc_uref[1] : !uref -> !muref
            %nonnull = reuse_ir.nullable.nonnull (%rcnext : !urclist) : !nullable
            reuse_ir.mref.assign %nonnull to %acc_muref : !nullable, !muref
            scf.yield %rcnext : !urclist
          }
          %res_uref = reuse_ir.rc.borrow %res : !urclist -> !uref
          %res_muref = reuse_ir.proj %res_uref[1] : !uref -> !muref
          %nonnull = reuse_ir.nullable.nonnull (%rchead : !urclist) : !nullable
          reuse_ir.mref.assign %nonnull to %res_muref : !nullable, !muref
          %frozen = reuse_ir.rc.freeze (%rchead : !urclist) : !frclist
          reuse_ir.region.yield %frozen : !frclist
        } : !frclist
        %2 = reuse_ir.rc.borrow %1 : !frclist -> !fref
        func.call @opaque(%2) : (!fref) -> ()
        reuse_ir.rc.release (%1 : !frclist)
        return
    }
}
