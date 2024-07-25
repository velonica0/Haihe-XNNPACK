// Auto-generated file. Do not edit!
//   Template: src/f32-vbinary/vopc-rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2023 SiFive, Inc.
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
#include <xnnpack/vbinary.h>


void xnn_f32_vminc_ukernel__rvv_u8v(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float b = *input_b;
  size_t n = batch >> 2;

  do {
    size_t vl = __riscv_vsetvl_e32m8(n);
    n -= vl;
    vfloat32m8_t va = __riscv_vle32_v_f32m8(input_a, vl);
    input_a += vl;
    vfloat32m8_t vacc = __riscv_vfmin_vf_f32m8(va, b, vl);
    __riscv_vse32_v_f32m8(output, vacc, vl);
    output += vl;
  } while (n > 0);
}
