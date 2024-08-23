// RUN: %reuse-opt %s -convert-reuse-ir-to-llvm | %FileCheck %s
!mref = !reuse_ir.mref<f128, nonatomic>
!inner = !reuse_ir.composite<i32, i32, !mref>
!test = !reuse_ir.composite<!inner, i32>
!array = !reuse_ir.array<!test, 16, 16>
module @test attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"} 
{
    func.func @array_projection(%0: !reuse_ir.rc<!array, nonatomic, frozen>) -> !reuse_ir.nullable<!reuse_ir.rc<f128, nonatomic, frozen>> {
        %1 = reuse_ir.borrow %0 : 
            !reuse_ir.rc<!array, nonatomic, frozen>
            -> !reuse_ir.ref<!array, frozen>
        // CHECK: %[[REG0:[0-9a-z]+]] = llvm.getelementptr %{{[0-9a-z]+}}[13] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<16 x struct<(struct<(i32, i32, ptr)>, i32, array<4 x i8>)>>
        %2 = reuse_ir.proj %1[13] : 
            !reuse_ir.ref<!array, frozen> -> !reuse_ir.ref<!reuse_ir.array<!test, 16>, frozen>
        // CHECK: %[[REG1:[0-9a-z]+]] = llvm.getelementptr %[[REG0]][7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(i32, i32, ptr)>, i32, array<4 x i8>)>
        %3 = reuse_ir.proj %2[7] : 
            !reuse_ir.ref<!reuse_ir.array<!test, 16>, frozen> -> !reuse_ir.ref<!test, frozen>
        // CHECK: %[[REG2:[0-9a-z]+]] = llvm.getelementptr %[[REG1]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(i32, i32, ptr)>, i32, array<4 x i8>)>
        %4 = reuse_ir.proj %3[0] : 
            !reuse_ir.ref<!test, frozen> -> !reuse_ir.ref<!inner, frozen>
        // CHECK: %[[REG3:[0-9a-z]+]] = llvm.getelementptr %[[REG2]][0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i32, i32, ptr)>
        %5 = reuse_ir.proj %4[2] : 
            !reuse_ir.ref<!inner, frozen> -> !reuse_ir.ref<!mref, frozen>
        // CHECK: %{{[0-9a-z]+}} = llvm.load %[[REG3]] {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
        %6 = reuse_ir.load %5 : 
            !reuse_ir.ref<!mref, frozen> -> !reuse_ir.nullable<!reuse_ir.rc<f128, nonatomic, frozen>>
        return %6 : !reuse_ir.nullable<!reuse_ir.rc<f128, nonatomic, frozen>>
    }
}
