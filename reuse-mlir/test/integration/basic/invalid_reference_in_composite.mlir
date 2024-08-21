// RUN: %not %reuse-opt %s 2>&1 | %FileCheck %s
// CHECK: error: cannot have a reference type in a composite type
!test = !reuse_ir.composite<!reuse_ir.ref<i32, nonfreezing>, i32>
