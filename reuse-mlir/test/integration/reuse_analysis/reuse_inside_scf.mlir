// RUN: %reuse-opt %s -reuse-ir-token-reuse | %FileCheck %s
!rc64 = !reuse_ir.rc<i64, nonatomic, nonfreezing>
!rc64x2 = !reuse_ir.rc<!reuse_ir.composite<"test" {i64, i64}>, nonatomic, nonfreezing>
!i64token = !reuse_ir.token<size: 16, alignment: 8>
module @test attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"} {
    // CHECK: func.func @reuse(%[[regarg0:[a-z0-9]+]]: !reuse_ir.rc<i64, nonatomic, nonfreezing>, %[[regarg1:[a-z0-9]+]]: i1, %[[regarg2:[a-z0-9]+]]: i64, %[[regarg3:[a-z0-9]+]]: i64) -> !reuse_ir.rc<i64, nonatomic, nonfreezing> {
    // CHECK:  %[[reg0:[a-z0-9]+]] = reuse_ir.rc.release(%[[regarg0]] : <i64, nonatomic, nonfreezing>) : !reuse_ir.nullable<!reuse_ir.token<size : 16, alignment : 8>>
    // CHECK:  %[[reg1:[a-z0-9]+]] = scf.if %[[regarg1]] -> (!reuse_ir.rc<i64, nonatomic, nonfreezing>) {
    // CHECK:    %[[reg2:[a-z0-9]+]] = reuse_ir.token.ensure(%[[reg0]] : <!reuse_ir.token<size : 16, alignment : 8>>) : <size : 16, alignment : 8>
    // CHECK:    %[[reg3:[a-z0-9]+]] = reuse_ir.rc.create value(%[[regarg2]]) token(%[[reg2]]) : (i64, !reuse_ir.token<size : 16, alignment : 8>) -> !reuse_ir.rc<i64, nonatomic, nonfreezing>
    // CHECK:    scf.yield %[[reg3]] : !reuse_ir.rc<i64, nonatomic, nonfreezing>
    // CHECK:  } else {
    // CHECK:    %[[reg2:[a-z0-9]+]] = reuse_ir.token.ensure(%[[reg0]] : <!reuse_ir.token<size : 16, alignment : 8>>) : <size : 16, alignment : 8>
    // CHECK:    %[[reg3:[a-z0-9]+]] = reuse_ir.rc.create value(%[[regarg3]]) token(%[[reg2]]) : (i64, !reuse_ir.token<size : 16, alignment : 8>) -> !reuse_ir.rc<i64, nonatomic, nonfreezing>
    // CHECK:    scf.yield %[[reg3]] : !reuse_ir.rc<i64, nonatomic, nonfreezing>
    // CHECK:  }
    // CHECK:  return %[[reg1]] : !reuse_ir.rc<i64, nonatomic, nonfreezing>
    // CHECK: }
    func.func @reuse(%0: !rc64, %1: i1, %x: i64, %y: i64) -> !rc64 {
        %2 = reuse_ir.rc.release (%0 : !rc64) : !reuse_ir.nullable<!reuse_ir.token<size: 16, alignment: 8>>
        %3 = scf.if %1 -> !rc64 {
          %tk = reuse_ir.token.alloc : !i64token
          %4 = reuse_ir.rc.create value(%x) token(%tk) : (i64, !i64token) -> !rc64
          scf.yield %4 : !rc64  
        } else {
          %tk = reuse_ir.token.alloc : !i64token
          %5 = reuse_ir.rc.create value(%y) token(%tk) : (i64, !i64token) -> !rc64
          scf.yield %5 : !rc64  
        }
        return %3 : !rc64
    }

    func.func private @opaque(%0: !rc64)
    // CHECK: func.func @partial(%[[regarg0:[a-z0-9]+]]: !reuse_ir.rc<i64, nonatomic, nonfreezing>, %[[regarg1:[a-z0-9]+]]: i1, %[[regarg2:[a-z0-9]+]]: i64) {
    // CHECK:  %[[reg0:[a-z0-9]+]] = reuse_ir.rc.release(%[[regarg0]] : <i64, nonatomic, nonfreezing>) : !reuse_ir.nullable<!reuse_ir.token<size : 16, alignment : 8>>
    // CHECK:  scf.if %[[regarg1]] {
    // CHECK:    %[[reg1:[a-z0-9]+]] = reuse_ir.token.ensure(%[[reg0]] : <!reuse_ir.token<size : 16, alignment : 8>>) : <size : 16, alignment : 8>
    // CHECK:    %[[reg2:[a-z0-9]+]] = reuse_ir.rc.create value(%[[regarg2]]) token(%[[reg1]]) : (i64, !reuse_ir.token<size : 16, alignment : 8>) -> !reuse_ir.rc<i64, nonatomic, nonfreezing>
    // CHECK:    func.call @opaque(%[[reg2]]) : (!reuse_ir.rc<i64, nonatomic, nonfreezing>) -> ()
    // CHECK:  } else {
    // CHECK:    reuse_ir.token.free(%[[reg0]] : !reuse_ir.nullable<!reuse_ir.token<size : 16, alignment : 8>>)
    // CHECK:  }
    // CHECK:  return
    // CHECK: }
    func.func @partial(%0: !rc64, %1: i1, %x: i64) {
        %2 = reuse_ir.rc.release (%0 : !rc64) : !reuse_ir.nullable<!reuse_ir.token<size: 16, alignment: 8>>
        scf.if %1 {
          %tk = reuse_ir.token.alloc : !i64token
          %5 = reuse_ir.rc.create value(%x) token(%tk) : (i64, !i64token) -> !rc64
          func.call @opaque(%5) : (!rc64) -> ()
          scf.yield
        } else {
          scf.yield 
        }
        return
    }
}
