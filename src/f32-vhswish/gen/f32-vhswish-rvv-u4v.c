// Auto-generated file. Do not edit!
//   Template: src/f32-vhswish/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#ifdef CORE_C920
    #include "riscv_v_071_fix.h"
    printf("Core C920 detected\n");
#elif CORE_K1
    #include <riscv_vector.h>
    printf("Core K1 detected\n");
#endif

#include <xnnpack/common.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/vunary.h>


void xnn_f32_vhswish_ukernel__rvv_u4v(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_hswish_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float vsixth = params->scalar.sixth;
  const float vthree = params->scalar.three;
  const float vsix = params->scalar.six;
  const float vzero = 0.0f;
  assert(vthree == 3.0f);
  assert(vsix == 6.0f);

  batch >>= XNN_LOG2_SIZEOF_FLOAT;
  do {
    const size_t n = __riscv_vsetvl_e32m4(batch);
    vfloat32m4_t vx = __riscv_vle32_v_f32m4(input, n);
    input += n;
    vfloat32m4_t vacc = __riscv_vfadd_vf_f32m4(vx, vthree, n);
    vx = __riscv_vfmul_vf_f32m4(vx, vsixth, n);
    vacc = __riscv_vfmax_vf_f32m4(vacc, vzero, n);
    vacc = __riscv_vfmin_vf_f32m4(vacc, vsix, n);
    vacc = __riscv_vfmul_vv_f32m4(vacc, vx, n);
    __riscv_vse32_v_f32m4(output, vacc, n);
    output += n;

    batch -= n;
  } while (batch != 0);
}
