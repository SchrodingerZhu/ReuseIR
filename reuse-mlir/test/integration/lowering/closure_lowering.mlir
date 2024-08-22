// RUN: %reuse-opt %s -reuse-ir-closure-outlining | %FileCheck %s --check-prefix=CHECK-OUTLINING
// RUN: %reuse-opt %s -reuse-ir-closure-outlining -convert-reuse-ir-to-llvm | %FileCheck %s --check-prefix=CHECK-LOWERING
!closure = !reuse_ir.closure<(i32, i128) -> i128>
module @test attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"} 
{
    // CHECK-OUTLINING: func.func private @closure_test$$lambda0$$func(%arg0: !reuse_ir.ref<!reuse_ir.composite<i32, i128>, nonfreezing>) -> i128 {
    // CHECK-OUTLINING:   %0 = reuse_ir.proj %arg0[0] : <!reuse_ir.composite<i32, i128>, nonfreezing> -> <i32, nonfreezing>
    // CHECK-OUTLINING:   %1 = reuse_ir.load %0 : <i32, nonfreezing> -> i32
    // CHECK-OUTLINING:   %2 = reuse_ir.proj %arg0[1] : <!reuse_ir.composite<i32, i128>, nonfreezing> -> <i128, nonfreezing>
    // CHECK-OUTLINING:   %3 = reuse_ir.load %2 : <i128, nonfreezing> -> i128
    // CHECK-OUTLINING:   %4 = arith.extui %1 : i32 to i128
    // CHECK-OUTLINING:   %5 = arith.addi %4, %3 : i128
    // CHECK-OUTLINING:   return %5 : i128
    // CHECK-OUTLINING: }
    // TODO: check drop
    // CHECK-OUTLINING: reuse_ir.closure.vtable @closure_test$$lambda0$$vtable{closure_type : !reuse_ir.closure<(i32, i128) -> i128>, func : @closure_test$$lambda0$$func, clone : @closure_test$$lambda0$$clone, drop : @closure_test$$lambda0$$drop}
    // CHECK-OUTLINING: func.func @closure_test() -> !reuse_ir.closure<(i32, i128) -> i128> {
    // CHECK-OUTLINING:   %0 = reuse_ir.alloc : <size : 32, alignment : 16>
    // CHECK-OUTLINING:   %1 = reuse_ir.closure.assemble vtable(@closure_test$$lambda0$$vtable) argpack(%0 : !reuse_ir.token<size : 32, alignment : 16>) : <(i32, i128) -> i128>
    // CHECK-OUTLINING:   return %1 : !reuse_ir.closure<(i32, i128) -> i128>
    // CHECK-OUTLINING: }


    // CHECK-LOWERING: llvm.func @closure_test$$lambda0$$func(%arg0: !llvm.ptr) -> i128 attributes {sym_visibility = "private"} {
    // CHECK-LOWERING:   %0 = llvm.getelementptr %arg0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, array<12 x i8>, i128)>
    // CHECK-LOWERING:   %1 = llvm.load %0 {alignment = 4 : i64} : !llvm.ptr -> i32
    // CHECK-LOWERING:   %2 = llvm.getelementptr %arg0[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, array<12 x i8>, i128)>
    // CHECK-LOWERING:   %3 = llvm.load %2 {alignment = 16 : i64} : !llvm.ptr -> i128
    // CHECK-LOWERING:   %4 = arith.extui %1 : i32 to i128
    // CHECK-LOWERING:   %5 = arith.addi %4, %3 : i128
    // CHECK-LOWERING:   llvm.return %5 : i128
    // CHECK-LOWERING: }
    // TODO: llvm.func @closure_test$$lambda0$$clone(!llvm.ptr, i64) -> !llvm.ptr attributes {sym_visibility = "private"}
    // TODO: llvm.func @closure_test$$lambda0$$drop(!llvm.ptr, i64) attributes {sym_visibility = "private"}
    // CHECK-LOWERING: llvm.mlir.global internal constant @closure_test$$lambda0$$vtable() {addr_space = 0 : i32, alignment = 8 : i64, dso_local} : !llvm.struct<(ptr, ptr, ptr)> {
    // CHECK-LOWERING:   %0 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, ptr)>
    // CHECK-LOWERING:   %1 = llvm.mlir.addressof @closure_test$$lambda0$$func : !llvm.ptr
    // CHECK-LOWERING:   %2 = llvm.insertvalue %1, %0[0] : !llvm.struct<(ptr, ptr, ptr)> 
    // CHECK-LOWERING:   %3 = llvm.mlir.addressof @closure_test$$lambda0$$clone : !llvm.ptr
    // CHECK-LOWERING:   %4 = llvm.insertvalue %3, %2[1] : !llvm.struct<(ptr, ptr, ptr)> 
    // CHECK-LOWERING:   %5 = llvm.mlir.addressof @closure_test$$lambda0$$drop : !llvm.ptr
    // CHECK-LOWERING:   %6 = llvm.insertvalue %5, %4[2] : !llvm.struct<(ptr, ptr, ptr)> 
    // CHECK-LOWERING:   llvm.return %6 : !llvm.struct<(ptr, ptr, ptr)>
    // CHECK-LOWERING: }
    // CHECK-LOWERING: llvm.func @closure_test() -> !llvm.struct<(ptr, ptr, i64)> {
    // CHECK-LOWERING:   %0 = llvm.mlir.constant(32 : i64) : i64
    // CHECK-LOWERING:   %1 = llvm.mlir.constant(16 : i64) : i64
    // CHECK-LOWERING:   %2 = llvm.call @__reuse_ir_alloc(%0, %1) : (i64, i64) -> !llvm.ptr
    // CHECK-LOWERING:   %3 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64)>
    // CHECK-LOWERING:   %4 = llvm.mlir.constant(0 : i64) : i64
    // CHECK-LOWERING:   %5 = llvm.mlir.addressof @closure_test$$lambda0$$vtable : !llvm.ptr
    // CHECK-LOWERING:   %6 = llvm.insertvalue %5, %3[0] : !llvm.struct<(ptr, ptr, i64)> 
    // CHECK-LOWERING:   %7 = llvm.insertvalue %2, %6[1] : !llvm.struct<(ptr, ptr, i64)> 
    // CHECK-LOWERING:   %8 = llvm.insertvalue %4, %7[2] : !llvm.struct<(ptr, ptr, i64)> 
    // CHECK-LOWERING:   llvm.return %8 : !llvm.struct<(ptr, ptr, i64)>
    // CHECK-LOWERING: }
    func.func @closure_test() -> !closure {
        %1 = reuse_ir.closure.new {
            ^bb(%arg0: i32, %arg1: i128):
                %2 = arith.extui %arg0 : i32 to i128
                %3 = arith.addi %2, %arg1 : i128
                reuse_ir.closure.yield %3 : i128
        } : !closure
        return %1 : !closure
    }
}
