// RUN: %reuse-opt %s
!ilist = !reuse_ir.union<"list" incomplete>
!rclist = !reuse_ir.rc<!ilist, nonatomic, nonfreezing>
!cons = !reuse_ir.composite<"list::cons" {i32, !rclist}>
!nil = !reuse_ir.composite<"list::nil" {}>
!list = !reuse_ir.union<"list" {!cons, !nil}>
!reflist = !reuse_ir.ref<!ilist, nonfreezing>
!list_token = !reuse_ir.token<size: 32, alignment: 8>
!nullable = !reuse_ir.nullable<!list_token>
!refcons = !reuse_ir.ref<!cons, nonfreezing>
!refi32 = !reuse_ir.ref<i32, nonfreezing>
!refrc = !reuse_ir.ref<!rclist, nonfreezing>
module @test  attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"} {
    func.func @reverse(%list: !rclist, %acc: !rclist) -> !rclist {
        %ref = reuse_ir.rc.borrow %list : !rclist -> !reflist
        %tag = reuse_ir.union.get_tag %ref : !reflist -> index
        %res = scf.index_switch %tag -> !rclist 
        case 0 {
            %inner = reuse_ir.union.inspect %ref [0] : !reflist -> !refcons
            %refhead = reuse_ir.proj %inner [0] : !refcons -> !refi32
            %head = reuse_ir.load %refhead : !refi32 -> i32
            %reftail = reuse_ir.proj %inner [1] : !refcons -> !refrc
            %tail = reuse_ir.load %reftail : !refrc -> !rclist
            reuse_ir.rc.acquire (%tail : !rclist)
            %tk = reuse_ir.rc.release (%list : !rclist) : !nullable
            %cons = reuse_ir.composite.assemble (%head, %acc) : (i32, !rclist) -> !cons
            %next = reuse_ir.union.assemble (0, %cons) : (!cons) -> !list
            %res = reuse_ir.rc.create value(%next) : (!list) -> !rclist
            %recusive = func.call @reverse(%tail, %res) : (!rclist, !rclist) -> !rclist
            scf.yield %recusive : !rclist
        }
        case 1 {
            reuse_ir.union.inspect %ref [1] : !reflist
            %x = reuse_ir.rc.release (%list : !rclist) : !nullable
            reuse_ir.token.free (%x : !nullable)
            scf.yield %acc : !rclist
        }
        default {
            %y = reuse_ir.unreachable : !rclist
            scf.yield %y : !rclist
        }
        return %res : !rclist
    }
}
