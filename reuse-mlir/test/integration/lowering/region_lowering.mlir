// RUN: %reuse-opt %s
!urc = !reuse_ir.rc<i64, nonatomic, unfrozen>
!frc = !reuse_ir.rc<i64, nonatomic, frozen>
!mref = !reuse_ir.mref<i64, nonatomic>
!rci64 = !reuse_ir.rc<i64, nonatomic, nonfreezing>
!struct = !reuse_ir.composite<"test" {!mref, !mref, !rci64}>
!usrc = !reuse_ir.rc<!struct, nonatomic, unfrozen>
!fsrc = !reuse_ir.rc<!struct, nonatomic, frozen>
!nullable = !reuse_ir.nullable<!urc>
!i64token = !reuse_ir.token<size: 32, alignment: 8>
!stoken = !reuse_ir.token<size: 48, alignment: 8>
module @test attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"} {
    func.func @rc_release(%x : i64, %y : i64, %z: i64) -> !fsrc {
        %1 = reuse_ir.region.run {
          ^bb(%ctx : !reuse_ir.region_ctx):
             %null = reuse_ir.nullable.null : !nullable
             %token = reuse_ir.token.alloc : !i64token 
             %urc_ = reuse_ir.rc.create value(%x) token(%token) region(%ctx) : (i64, !i64token, !reuse_ir.region_ctx) -> !urc
             %urc = reuse_ir.nullable.nonnull (%urc_ : !urc) : !nullable
             %tk = reuse_ir.token.alloc : !reuse_ir.token<size: 16, alignment: 8>
             %i64rc = reuse_ir.rc.create value(%z) token(%tk) : (i64, !reuse_ir.token<size: 16, alignment: 8>) -> !rci64
             %struct = reuse_ir.composite.assemble (%urc, %null, %i64rc) : (!nullable, !nullable, !rci64) -> !struct
             %stoken = reuse_ir.token.alloc : !stoken
             %usrc = reuse_ir.rc.create value(%struct) token(%stoken) region(%ctx) : (!struct, !stoken, !reuse_ir.region_ctx) -> !usrc
             %borrowed = reuse_ir.rc.borrow %usrc : !usrc -> !reuse_ir.ref<!struct, unfrozen>
             %proj = reuse_ir.proj %borrowed[1] : !reuse_ir.ref<!struct, unfrozen> -> !reuse_ir.ref<!mref, unfrozen>
             %token1 = reuse_ir.token.alloc : !i64token 
             %urc2_ = reuse_ir.rc.create value(%y) token(%token1) region(%ctx) : (i64, !i64token, !reuse_ir.region_ctx) -> !urc
             %urc2 = reuse_ir.nullable.nonnull (%urc2_ : !urc) : !nullable
             reuse_ir.mref.assign %urc2 to %proj : !nullable, !reuse_ir.ref<!mref, unfrozen>
             %fsrc = reuse_ir.rc.freeze (%usrc : !usrc) : !fsrc
             reuse_ir.region.yield %fsrc : !fsrc
        } : !fsrc
        return %1 : !fsrc
    }
}
