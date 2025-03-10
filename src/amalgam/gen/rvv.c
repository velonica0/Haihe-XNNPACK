// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <string.h>

//#include <riscv_vector.h>
#include "riscv_v_071_fix.h"
#include <fp16/fp16.h>

#include <xnnpack/common.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/math.h>
#include <xnnpack/raddstoreexpminusmax.h>
#include <xnnpack/vbinary.h>
#include <xnnpack/vunary.h>

#include <xnnpack/rvv_mathfun_fp16.h>
#include <xnnpack/rvv_mathfun_fp32.h>

static inline vfloat32m4_t eval_poly_horner(vfloat32m4_t x,
                                                  float c6, float c5,
                                                  float c4, float c3, float c2,
                                                  float c1, float c0, size_t vl) {
  vfloat32m4_t z;
  vfloat32m4_t y = __riscv_vfmv_v_f_f32m4(c5, vl);
  y = __riscv_vfmacc_vf_f32m4(y, c6, x, vl);

  z = __riscv_vfmv_v_f_f32m4(c4, vl);
  y = __riscv_vfmadd_vv_f32m4(y, x, z, vl);

  z = __riscv_vfmv_v_f_f32m4(c3, vl);
  y = __riscv_vfmadd_vv_f32m4(y, x, z, vl);

  z = __riscv_vfmv_v_f_f32m4(c2, vl);
  y = __riscv_vfmadd_vv_f32m4(y, x, z, vl);

  z = __riscv_vfmv_v_f_f32m4(c1, vl);
  y = __riscv_vfmadd_vv_f32m4(y, x, z, vl);

  z = __riscv_vfmv_v_f_f32m4(c0, vl);
  y = __riscv_vfmadd_vv_f32m4(y, x, z, vl);
  return y;
}

/// @brief Computes the exponential function on vector of float32 values with a
/// 1-ULP error bound in the range [-87, 0]. Smaller inputs are flushed to
/// exp(-0x1.5d589ep6f) ~= 0x1.6a0a64p-127f while the result is undefined for
/// inputs greater than zero as well as NaNs.
///
/// This function is intended for use in computing softmax, whose inputs are
/// pre-normalized by subtracting the maximum, resulting in inputs in (-inf, 0).
/// One of these inputs will contribute exp(0) = 1 to the final sum, so any
/// inputs flushed upwards to -0x1.5d589ep6f and thus contributing at most
/// 0x1.6a0a64p-127f to the total, will not result of softmax unless at least
/// ~2^100 of them are summed in ascending order.
///
/// Exploitation of these properties results in a faster exponential by avoiding
/// the need to handle edge cases that arise from very large or small exponents.
///
/// @param[in] x Input vector of float32 values
/// @param[in] vl Length of vector x
/// @return Result of applying softexp() to elements of x
static inline vfloat32m4_t softexp_f32m4(
    vfloat32m4_t x, size_t vl,
    const union xnn_f32_expminus_params params[restrict XNN_MIN_ELEMENTS(1)]) {
  // Ensure that q = RN(x/log(2)) >= e_min, so that 2^q can be computed safely
  // with a simple shift into the exponent field.
  // xmin = round(-126.5 * log(2), single, RU) ~ -87.68311309814453125

  const float xmin = params->rvv_rr2_p6.x_min;
  const float r_ln2f = params->rvv_rr2_p6.log2e;
  const float l2uf = params->rvv_rr2_p6.ln2_hi;
  const float l2lf = params->rvv_rr2_p6.ln2_lo;
  const float c6 = params->rvv_rr2_p6.c6;
  const float c5 = params->rvv_rr2_p6.c5;
  const float c4 = params->rvv_rr2_p6.c4;
  const float c3 = params->rvv_rr2_p6.c3;
  const float c2 = params->rvv_rr2_p6.c2;

  // const float xmin = -0x1.5ebb82p6;
  x = __riscv_vfmax_vf_f32m4(x, xmin, vl);

  // 0. Reduction x = s * q ln(2)
  // const float r_ln2f = 0x1.715476p0f;  // single(1/log(2));
  // const float l2uf = 0x1.62e4p-1f;     // round(log(2), 24-8, RN);
  // const float l2lf = 0x1.7f7d1cp-20f;  // round(log(2) - l2uf, single, RN);
  vfloat32m4_t v = __riscv_vfmul_vf_f32m4(x, r_ln2f, vl);

  vint16m2_t q = __riscv_vfncvt_x_f_w_i16m2(v, vl);
  vfloat32m4_t z = __riscv_vfwcvt_f_x_v_f32m4(q, vl);

  // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
  vfloat32m4_t s = __riscv_vfnmsac_vf_f32m4(x, l2uf, z, vl);
  s = __riscv_vfnmsac_vf_f32m4(s, l2lf, z, vl);

  // 1. Approximate e^s by degree-6 polynomial approximation
  vfloat32m4_t u = eval_poly_horner(s, c6, c5, c4, c3, c2, 1.0f, 1.0f, vl);

  // 2. Reconstruction: compute u = u*2^q
  const int16_t p = (24 - 1);
  const int16_t bias = (128 - 1);
  vint32m4_t qw = __riscv_vwadd_vx_i32m4(q, bias, vl);
  vint32m4_t qq = __riscv_vsll_vx_i32m4(qw, p, vl);
  vfloat32m4_t qf = __riscv_vreinterpret_v_i32m4_f32m4(qq);
  u = __riscv_vfmul_vv_f32m4(u, qf, vl);
  return u;
}

void xnn_f32_raddstoreexpminusmax_ukernel__rvv_rr2_p6_u4v(
    size_t batch,
    const float* input,
    const float* max,
    float* output,
    float* sum,
    const union xnn_f32_expminus_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(max != NULL);
  assert(output != NULL);
  assert(sum != NULL);

  size_t n = batch >> 2;
  size_t avl = n;
  size_t vl = __riscv_vsetvl_e32m4(n);

  vfloat32m4_t vsum = __riscv_vfmv_v_f_f32m4(0.0f, vl);
  do {
    vl = __riscv_vsetvl_e32m4(avl);
    avl -= vl;
    vfloat32m4_t vx = __riscv_vle32_v_f32m4(input, vl);
    vx = __riscv_vfsub_vf_f32m4(vx, *max, vl);
    input += vl;
    vfloat32m4_t vexp = softexp_f32m4(vx, vl, params);
    __riscv_vse32_v_f32m4(output, vexp, vl);
    output += vl;
    vsum = __riscv_vfadd_vv_f32m4_tu(vsum, vsum, vexp, vl);
  } while(avl > 0);

  //vfloat32m1_t v0 = __riscv_vfmv_s_f_f32m1(0.0f, 1);
  //*sum = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredusum_vs_f32m4_f32m1(vsum, v0, n));
  vfloat32m1_t v0, v1;
  v0 = vfmv_s_f_f32m1(v0, 0.0f, 1);
  v1 = vfredosum_vs_f32m4_f32m1(v1, vsum, v0, n);
  *sum = __riscv_vfmv_f_s_f32m1_f32(v1);
}

void xnn_f32_rmax_ukernel__rvv_u8v(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  size_t N = batch >> 2;
  size_t avl;
  size_t vl = __riscv_vsetvl_e32m8(N);

  vfloat32m8_t t0 = __riscv_vle32_v_f32m8(input, vl);
  input += vl;

  for (avl = N - vl; avl; avl -= vl, input += vl) {
    vl = __riscv_vsetvl_e32m8(avl);
    vfloat32m8_t vec = __riscv_vle32_v_f32m8(input, vl);
    t0 = __riscv_vfmax_vv_f32m8_tu(t0, t0, vec, vl);
  }

  //vfloat32m1_t fmax = __riscv_vfmv_s_f_f32m1(-INFINITY, 1);
  //output[0] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m8_f32m1(t0, fmax, N));
  vfloat32m1_t fmax, v0;
  fmax = vfmv_s_f_f32m1(fmax, -INFINITY, 1);
  v0 = vfredmax_vs_f32m8_f32m1(v0, t0, fmax, N);
  output[0] = __riscv_vfmv_f_s_f32m1_f32(v0);
}

void xnn_f32_rminmax_ukernel__rvv_u8v(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  size_t N = batch >> 2;
  size_t avl;
  size_t vl = __riscv_vsetvl_e32m8(N);

  vfloat32m8_t t0 = __riscv_vle32_v_f32m8(input, vl);
  input += vl;
  vfloat32m8_t t1 = __riscv_vmv_v_v_f32m8(t0, vl);

  for (avl = N - vl; avl; avl -= vl, input += vl) {
    vl = __riscv_vsetvl_e32m8(avl);
    vfloat32m8_t vec = __riscv_vle32_v_f32m8(input, vl);
    t0 = __riscv_vfmin_vv_f32m8_tu(t0, t0, vec, vl);
    t1 = __riscv_vfmax_vv_f32m8_tu(t1, t1, vec, vl);
  }

  //vfloat32m1_t fmin = __riscv_vfmv_s_f_f32m1(INFINITY, 1);
  //vfloat32m1_t fmax = __riscv_vfmv_s_f_f32m1(-INFINITY, 1);
  //output[0] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmin_vs_f32m8_f32m1(t0, fmin, N));
  //output[1] = __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m8_f32m1(t1, fmax, N));
  vfloat32m1_t fmin, fmax, v0, v1;
  fmin = vfmv_s_f_f32m1(fmin, INFINITY, 1);
  fmax = vfmv_s_f_f32m1(fmax, -INFINITY, 1);
  v0 = vfredmin_vs_f32m8_f32m1(v0, t0, fmin, N);
  v1 = vfredmax_vs_f32m8_f32m1(v1, t1, fmax, N);
  output[0] = __riscv_vfmv_f_s_f32m1_f32(v0);
  output[1] = __riscv_vfmv_f_s_f32m1_f32(v1);
}

void xnn_qs8_vmul_minmax_fp32_ukernel__rvv_u2v(
    size_t batch,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_mul_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const int32_t a_zero_point = params->fp32_scalar.a_zero_point;
  const int32_t b_zero_point = params->fp32_scalar.b_zero_point;
  const float scale = params->fp32_scalar.scale;
  const float output_min_less_zero_point = params->fp32_scalar.output_min_less_zero_point;
  const float output_max_less_zero_point = params->fp32_scalar.output_max_less_zero_point;
  const float magic_bias = params->fp32_scalar.magic_bias;
  const int32_t magic_bias_less_output_zero_point = params->fp32_scalar.magic_bias_less_output_zero_point;

  do {
    const size_t n = __riscv_vsetvl_e8m2(batch);

    vint8m2_t in_a_i8v = __riscv_vle8_v_i8m2(input_a, n); input_a += n;
    vint8m2_t in_b_i8v = __riscv_vle8_v_i8m2(input_b, n); input_b += n;
    vint16m4_t a_i16v = __riscv_vwsub_vx_i16m4(in_a_i8v, a_zero_point, n);
    vint16m4_t b_i16v = __riscv_vwsub_vx_i16m4(in_b_i8v, b_zero_point, n);

    vint32m8_t acc_i32v = __riscv_vwmul_vv_i32m8(a_i16v, b_i16v, n);
    vfloat32m8_t acc_f32v = __riscv_vfcvt_f_x_v_f32m8(acc_i32v, n);
    acc_f32v = __riscv_vfmul_vf_f32m8(acc_f32v, scale, n);
    acc_f32v = __riscv_vfmin_vf_f32m8(__riscv_vfmax_vf_f32m8(acc_f32v, output_min_less_zero_point, n), output_max_less_zero_point, n);
    acc_f32v = __riscv_vfadd_vf_f32m8(acc_f32v, magic_bias, n);

    vint32m8_t out_i32v = __riscv_vfcvt_x_f_v_i32m8(acc_f32v, n);
    out_i32v = __riscv_vsub_vx_i32m8(out_i32v, magic_bias_less_output_zero_point, n);
    vint16m4_t out_i16v = __riscv_vncvt_x_x_w_i16m4(out_i32v, n);
    vint8m2_t out_i8v = __riscv_vncvt_x_x_w_i8m2(out_i16v, n);
    __riscv_vse8_v_i8m2(output, out_i8v, n); output += n;

    batch -= n;
  } while (batch != 0);
}

void xnn_qs8_vmulc_minmax_fp32_ukernel__rvv_u2v(
    size_t batch,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_mul_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const int32_t a_zero_point = params->fp32_scalar.a_zero_point;
  const float scale = params->fp32_scalar.scale;
  const float output_min_less_zero_point = params->fp32_scalar.output_min_less_zero_point;
  const float output_max_less_zero_point = params->fp32_scalar.output_max_less_zero_point;
  const float magic_bias = params->fp32_scalar.magic_bias;
  const int32_t magic_bias_less_output_zero_point = params->fp32_scalar.magic_bias_less_output_zero_point;
  const int32_t vb = (int32_t) *input_b - params->fp32_scalar.b_zero_point;

  do {
    const size_t n = __riscv_vsetvl_e8m2(batch);

    vint8m2_t in_a_i8v = __riscv_vle8_v_i8m2(input_a, n); input_a += n;
    vint16m4_t a_i16v = __riscv_vwsub_vx_i16m4(in_a_i8v, a_zero_point, n);

    vint32m8_t acc_i32v = __riscv_vwmul_vx_i32m8(a_i16v, vb, n);
    vfloat32m8_t acc_f32v = __riscv_vfcvt_f_x_v_f32m8(acc_i32v, n);
    acc_f32v = __riscv_vfmul_vf_f32m8(acc_f32v, scale, n);
    acc_f32v = __riscv_vfmin_vf_f32m8(__riscv_vfmax_vf_f32m8(acc_f32v, output_min_less_zero_point, n), output_max_less_zero_point, n);
    acc_f32v = __riscv_vfadd_vf_f32m8(acc_f32v, magic_bias, n);

    vint32m8_t out_i32v = __riscv_vfcvt_x_f_v_i32m8(acc_f32v, n);
    out_i32v = __riscv_vsub_vx_i32m8(out_i32v, magic_bias_less_output_zero_point, n);
    vint16m4_t out_i16v = __riscv_vncvt_x_x_w_i16m4(out_i32v, n);
    vint8m2_t out_i8v = __riscv_vncvt_x_x_w_i8m2(out_i16v, n);
    __riscv_vse8_v_i8m2(output, out_i8v, n); output += n;

    batch -= n;
  } while (batch != 0);
}

void xnn_qu8_vmul_minmax_fp32_ukernel__rvv_u2v(
    size_t batch,
    const uint8_t* input_a,
    const uint8_t* input_b,
    uint8_t* output,
    const union xnn_qu8_mul_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const int32_t a_zero_point = params->fp32_scalar.a_zero_point;
  const int32_t b_zero_point = params->fp32_scalar.b_zero_point;
  const float scale = params->fp32_scalar.scale;
  const float output_min_less_zero_point = params->fp32_scalar.output_min_less_zero_point;
  const float output_max_less_zero_point = params->fp32_scalar.output_max_less_zero_point;
  const float magic_bias = params->fp32_scalar.magic_bias;
  const int32_t magic_bias_less_output_zero_point = params->fp32_scalar.magic_bias_less_output_zero_point;

  do {
    const size_t n = __riscv_vsetvl_e8m2(batch);

    vuint8m2_t in_a_u8v = __riscv_vle8_v_u8m2(input_a, n); input_a += n;
    vuint8m2_t in_b_u8v = __riscv_vle8_v_u8m2(input_b, n); input_b += n;
    vuint16m4_t a_u16v = __riscv_vwsubu_vx_u16m4(in_a_u8v, a_zero_point, n);
    vuint16m4_t b_u16v = __riscv_vwsubu_vx_u16m4(in_b_u8v, b_zero_point, n);
    vint16m4_t a_i16v = __riscv_vreinterpret_v_u16m4_i16m4(a_u16v);
    vint16m4_t b_i16v = __riscv_vreinterpret_v_u16m4_i16m4(b_u16v);

    vint32m8_t acc_i32v = __riscv_vwmul_vv_i32m8(a_i16v, b_i16v, n);
    vfloat32m8_t acc_f32v = __riscv_vfcvt_f_x_v_f32m8(acc_i32v, n);
    acc_f32v = __riscv_vfmul_vf_f32m8(acc_f32v, scale, n);
    acc_f32v = __riscv_vfmin_vf_f32m8(__riscv_vfmax_vf_f32m8(acc_f32v, output_min_less_zero_point, n), output_max_less_zero_point, n);
    acc_f32v = __riscv_vfadd_vf_f32m8(acc_f32v, magic_bias, n);

    vuint32m8_t out_u32v = __riscv_vfcvt_xu_f_v_u32m8(acc_f32v, n);
    out_u32v = __riscv_vsub_vx_u32m8(out_u32v, magic_bias_less_output_zero_point, n);
    vuint16m4_t out_u16v = __riscv_vncvt_x_x_w_u16m4(out_u32v, n);
    vuint8m2_t out_u8v = __riscv_vncvt_x_x_w_u8m2(out_u16v, n);
    __riscv_vse8_v_u8m2(output, out_u8v, n); output += n;

    batch -= n;
  } while (batch != 0);
}

void xnn_qu8_vmulc_minmax_fp32_ukernel__rvv_u2v(
    size_t batch,
    const uint8_t* input_a,
    const uint8_t* input_b,
    uint8_t* output,
    const union xnn_qu8_mul_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const int32_t a_zero_point = params->fp32_scalar.a_zero_point;
  const float scale = params->fp32_scalar.scale;
  const float output_min_less_zero_point = params->fp32_scalar.output_min_less_zero_point;
  const float output_max_less_zero_point = params->fp32_scalar.output_max_less_zero_point;
  const float magic_bias = params->fp32_scalar.magic_bias;
  const int32_t magic_bias_less_output_zero_point = params->fp32_scalar.magic_bias_less_output_zero_point;
  const int32_t vb = (int32_t) *input_b - params->fp32_scalar.b_zero_point;

  do {
    const size_t n = __riscv_vsetvl_e8m2(batch);

    vuint8m2_t in_a_u8v = __riscv_vle8_v_u8m2(input_a, n); input_a += n;
    vuint16m4_t a_u16v = __riscv_vwsubu_vx_u16m4(in_a_u8v, a_zero_point, n);
    vint16m4_t a_i16v = __riscv_vreinterpret_v_u16m4_i16m4(a_u16v);

    vint32m8_t acc_i32v = __riscv_vwmul_vx_i32m8(a_i16v, vb, n);
    vfloat32m8_t acc_f32v = __riscv_vfcvt_f_x_v_f32m8(acc_i32v, n);
    acc_f32v = __riscv_vfmul_vf_f32m8(acc_f32v, scale, n);
    acc_f32v = __riscv_vfmin_vf_f32m8(__riscv_vfmax_vf_f32m8(acc_f32v, output_min_less_zero_point, n), output_max_less_zero_point, n);
    acc_f32v = __riscv_vfadd_vf_f32m8(acc_f32v, magic_bias, n);

    vuint32m8_t out_u32v = __riscv_vfcvt_xu_f_v_u32m8(acc_f32v, n);
    out_u32v = __riscv_vsub_vx_u32m8(out_u32v, magic_bias_less_output_zero_point, n);
    vuint16m4_t out_u16v = __riscv_vncvt_x_x_w_u16m4(out_u32v, n);
    vuint8m2_t out_u8v = __riscv_vncvt_x_x_w_u8m2(out_u16v, n);
    __riscv_vse8_v_u8m2(output, out_u8v, n); output += n;

    batch -= n;
  } while (batch != 0);
}

void xnn_f32_vlrelu_ukernel__rvv_u8v(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
	assert(batch != 0);
	assert(batch % sizeof(float) == 0);
	assert(input != NULL);
	assert(output != NULL);

	const float vslope = params->scalar.slope;
	size_t size = batch / sizeof(float);

	do {
		const size_t n = vsetvl_e32m8(size);

		vfloat32m8_t in_u8v = vle32_v_f32m8(input, n);
		input += n;
		vbool4_t mask = vmflt_vf_f32m8_b4(in_u8v, .0f, n);
		vfloat32m8_t out_u8v = vfmul_vf_f32m8_m(mask, in_u8v, in_u8v, vslope, n);
		//vfloat32m8_t out_u8v = vfmax_vf_f32m8(in_u8v, .0f, n);
		vse32_v_f32m8(output, out_u8v, n);

		output += n;
		size -= n;
	} while (size > 0);
}


void xnn_f32_gemm_ukernel_1x4__rvv_u1v(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
	assert(mr != 0);
	assert(mr <= 1);
	assert(nc != 0);
	assert(kc != 0);
	assert(kc % sizeof(float) == 0);
	assert(a != NULL);
	assert(w != NULL);
	assert(c != NULL);

	const float* a0 = a;
	float* c0 = c;
	size_t kcl = kc / sizeof(float);

	do {
		size_t vl = vsetvl_e32m1(nc);
		vfloat32m1_t vacc = vle32_v_f32m1(w, 4);
		w += 4;
		for(size_t k = 0; k < kcl ; k++){
			vfloat32m1_t vw = vle32_v_f32m1(w, 4);
			w += 4;
			vacc = vfmacc_vf_f32m1(vacc, *a0, vw, 4);
			a0++;
		}
		vse32_v_f32m1(c0, vacc, vl);
		if(nc >= 4){
      		c0 = (float*) ((uintptr_t) c0 + cn_stride);
      		a0 = (const void*) ((uintptr_t) a0 - kc);
		}
		nc -= vl;
	} while (nc != 0);
}

//void xnn_f32_gemm_ukernel_2x4__rvv_u1v(
//        size_t mr,
//        size_t nc,
//        size_t kc,
//        const float* restrict a,
//        size_t a_stride,
//        const float* restrict w,
//        float* restrict c,
//        size_t cm_stride,
//        size_t cn_stride,
//        const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
//{
//    assert(mr != 0);
//    assert(mr <= 2); // max process 2 row
//    assert(nc != 0);
//    assert(kc != 0);
//    assert(kc % sizeof(float) == 0);
//    assert(a != NULL);
//    assert(w != NULL);
//    assert(c != NULL);
//
//    const float* a0 = a; // matrix a row 0 pointer
//    const float* a1 = a + a_stride; // matrix a row 1 pointer
//    float* c0 = c; // row 0 start pointer
//    float* c1 = c + cm_stride; // row 1 start pointer
//    size_t kcl = kc / sizeof(float);
//
//    do {
//        size_t vl = vsetvl_e32m1(nc);
//        vfloat32m1_t vacc0 = vfsub_vv_f32m1(vle32_v_f32m1(c0, vl), vle32_v_f32m1(c0, vl), vl); // 0th row count
//        vfloat32m1_t vacc1 = vfsub_vv_f32m1(vle32_v_f32m1(c1, vl), vle32_v_f32m1(c1, vl), vl); // 1st row count
//        w += vl;
//        for(size_t k = 0; k < kcl ; k++){
//            vfloat32m1_t va0 = vfmv_v_f_f32m1(*a0, vl); // load 0th row of matrix A
//            vfloat32m1_t va1 = vfmv_v_f_f32m1(*a1, vl); // load 1st row of matrix A
//            vfloat32m1_t vw = vle32_v_f32m1(w, vl); // load w
//            vacc0 = vfmacc_vv_f32m1(vacc0, va0, vw, vl); // update 0th row count
//            vacc1 = vfmacc_vv_f32m1(vacc1, va1, vw, vl); // update 1st row count
//            a0++;
//            a1++;
//            w += vl; // move matrix w pointer
//        }
//        vse32_v_f32m1(c0, vacc0, vl); // store 0th row result
//        vse32_v_f32m1(c1, vacc1, vl); // store 1st row result
//        c0 += cn_stride; // update 0th row matrix C pointer
//        c1 += cn_stride; // update 1st row matrix C pointer
//        a0 = a; // reset 0th row matrix A pointer
//        a1 = a + a_stride; // reset 1st row matrix A pointer
//        nc -= vl;
//
//    } while (nc != 0);
//}

void xnn_f32_gemm_ukernel_4x4__rvv_u1v(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(mr != 0);
    assert(mr <= 4); // max process 1 row
    assert(nc != 0);
    assert(kc != 0);
    assert(kc % sizeof(float) == 0);
    assert(a != NULL);
    assert(w != NULL);
    assert(c != NULL);

    const float* a0 = a;
    float* c0 = c;
    const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
    float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
    if XNN_UNPREDICTABLE(mr < 2) {
        a1 = a0;
        c1 = c0;
    }
    const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
    float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
    if XNN_UNPREDICTABLE(mr <= 2) {
        a2 = a1;
        c2 = c1;
    }
    const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
    float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
    if XNN_UNPREDICTABLE(mr != 4) {
        a3 = a2;
        c3 = c2;
    }

    size_t kcl = kc / sizeof(float);

    do {
        size_t vl = vsetvl_e32m1(nc); // vector length
        vfloat32m1_t vacc0 = vle32_v_f32m1(w, 4); // 1st row count
        vfloat32m1_t vacc1 = vle32_v_f32m1(w, 4); // 1st row count
        vfloat32m1_t vacc2 = vle32_v_f32m1(w, 4); // 1st row count
        vfloat32m1_t vacc3 = vle32_v_f32m1(w, 4); // 1st row count
        w += 4;
        for(size_t k = 0; k < kcl ; k++){
            vfloat32m1_t vw = vle32_v_f32m1(w, 4);
            w += 4;
            vacc0 = vfmacc_vf_f32m1(vacc0, *a0, vw, 4); // update 1st row count
            vacc1 = vfmacc_vf_f32m1(vacc1, *a1, vw, 4); // update 1st row count
            vacc2 = vfmacc_vf_f32m1(vacc2, *a2, vw, 4); // update 1st row count
            vacc3 = vfmacc_vf_f32m1(vacc3, *a3, vw, 4); // update 1st row count
            a0++;
            a1++;
            a2++;
            a3++;
        }
        vse32_v_f32m1(c0, vacc0, vl); // store 1st row result
        vse32_v_f32m1(c1, vacc1, vl); // store 1st row result
        vse32_v_f32m1(c2, vacc2, vl); // store 1st row result
        vse32_v_f32m1(c3, vacc3, vl); // store 1st row result
        if(nc >= 4){
            c0 = (float*) ((uintptr_t) c0 + cn_stride); // update 1st row matrix C pointer
            c1 = (float*) ((uintptr_t) c1 + cn_stride); // update 1st row matrix C pointer
            c2 = (float*) ((uintptr_t) c2 + cn_stride); // update 1st row matrix C pointer
            c3 = (float*) ((uintptr_t) c3 + cn_stride); // update 1st row matrix C pointer
            a0 = (const void*) ((uintptr_t) a0 - kc); // update 1st row matrix A pointer
            a1 = (const void*) ((uintptr_t) a1 - kc); // update 1st row matrix A pointer
            a2 = (const void*) ((uintptr_t) a2 - kc); // update 1st row matrix A pointer
            a3 = (const void*) ((uintptr_t) a3 - kc); // update 1st row matrix A pointer
        }
        nc -= vl;
    } while (nc != 0);
}

void xnn_f32_gemm_ukernel_1x8__rvv_u2v(
        size_t mr,
        size_t nc,
        size_t kc,
        const float* restrict a,
        size_t a_stride,
        const float* restrict w,
        float* restrict c,
        size_t cm_stride,
        size_t cn_stride,
        const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(mr != 0);
    assert(mr <= 1);
    assert(nc != 0);
    assert(kc != 0);
    assert(kc % sizeof(float) == 0);
    assert(a != NULL);
    assert(w != NULL);
    assert(c != NULL);

    const float* a0 = a;
    float* c0 = c;
    size_t kcl = kc / sizeof(float);

    do {
        size_t vl = vsetvl_e32m2(nc);
        vfloat32m2_t vacc = vle32_v_f32m2(w, 8);
        w += 8;
        for(size_t k = 0; k < kcl ; k++){
            vfloat32m2_t vw = vle32_v_f32m2(w, 8);
            w += 8;
            vacc = vfmacc_vf_f32m2(vacc, *a0, vw, 8);
            a0++;
        }
        vse32_v_f32m2(c0, vacc, vl);
        if(nc >= 4){
            c0 = (float*) ((uintptr_t) c0 + cn_stride);
            a0 = (const void*) ((uintptr_t) a0 - kc);
        }
        nc -= vl;
    } while (nc != 0);
}

void xnn_f32_gemm_ukernel_4x8__rvv_u2v(
        size_t mr,
        size_t nc,
        size_t kc,
        const float* restrict a,
        size_t a_stride,
        const float* restrict w,
        float* restrict c,
        size_t cm_stride,
        size_t cn_stride,
        const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(mr != 0);
    assert(mr <= 4); // max process 1 row
    assert(nc != 0);
    assert(kc != 0);
    assert(kc % sizeof(float) == 0);
    assert(a != NULL);
    assert(w != NULL);
    assert(c != NULL);

    const float* a0 = a;
    float* c0 = c;
    const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
    float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
    if XNN_UNPREDICTABLE(mr < 2) {
        a1 = a0;
        c1 = c0;
    }
    const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
    float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
    if XNN_UNPREDICTABLE(mr <= 2) {
        a2 = a1;
        c2 = c1;
    }
    const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
    float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
    if XNN_UNPREDICTABLE(mr != 4) {
        a3 = a2;
        c3 = c2;
    }

    size_t kcl = kc / sizeof(float);

    do {
        size_t vl = vsetvl_e32m2(nc); // vector length
        vfloat32m2_t vacc0 = vle32_v_f32m2(w, 8); // 1st row count
        vfloat32m2_t vacc1 = vle32_v_f32m2(w, 8); // 1st row count
        vfloat32m2_t vacc2 = vle32_v_f32m2(w, 8); // 1st row count
        vfloat32m2_t vacc3 = vle32_v_f32m2(w, 8); // 1st row count
        w += 8;
        for(size_t k = 0; k < kcl ; k++){
            vfloat32m2_t vw = vle32_v_f32m2(w, 8);
            w += 8;
            vacc0 = vfmacc_vf_f32m2(vacc0, *a0, vw, 8); // update 1st row count
            vacc1 = vfmacc_vf_f32m2(vacc1, *a1, vw, 8); // update 1st row count
            vacc2 = vfmacc_vf_f32m2(vacc2, *a2, vw, 8); // update 1st row count
            vacc3 = vfmacc_vf_f32m2(vacc3, *a3, vw, 8); // update 1st row count
            a0++;
            a1++;
            a2++;
            a3++;
        }
        vse32_v_f32m2(c0, vacc0, vl); // store 1st row result
        vse32_v_f32m2(c1, vacc1, vl); // store 1st row result
        vse32_v_f32m2(c2, vacc2, vl); // store 1st row result
        vse32_v_f32m2(c3, vacc3, vl); // store 1st row result
        if(nc >= 4){
            c0 = (float*) ((uintptr_t) c0 + cn_stride); // update 1st row matrix C pointer
            c1 = (float*) ((uintptr_t) c1 + cn_stride); // update 1st row matrix C pointer
            c2 = (float*) ((uintptr_t) c2 + cn_stride); // update 1st row matrix C pointer
            c3 = (float*) ((uintptr_t) c3 + cn_stride); // update 1st row matrix C pointer
            a0 = (const void*) ((uintptr_t) a0 - kc); // update 1st row matrix A pointer
            a1 = (const void*) ((uintptr_t) a1 - kc); // update 1st row matrix A pointer
            a2 = (const void*) ((uintptr_t) a2 - kc); // update 1st row matrix A pointer
            a3 = (const void*) ((uintptr_t) a3 - kc); // update 1st row matrix A pointer
        }
        nc -= vl;
    } while (nc != 0);

}

void xnn_f32_gemm_ukernel_1x16__rvv_u4v(
        size_t mr,
        size_t nc,
        size_t kc,
        const float* restrict a,
        size_t a_stride,
        const float* restrict w,
        float* restrict c,
        size_t cm_stride,
        size_t cn_stride,
        const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(mr != 0);
    assert(mr <= 1);
    assert(nc != 0);
    assert(kc != 0);
    assert(kc % sizeof(float) == 0);
    assert(a != NULL);
    assert(w != NULL);
    assert(c != NULL);

    const float* a0 = a;
    float* c0 = c;
    size_t kcl = kc / sizeof(float);

    do {
        size_t vl = vsetvl_e32m4(nc);
        vfloat32m4_t vacc = vle32_v_f32m4(w, 16);
        w += 16;
        for(size_t k = 0; k < kcl ; k++){
            vfloat32m4_t vw = vle32_v_f32m4(w, 16);
            w += 16;
            vacc = vfmacc_vf_f32m4(vacc, *a0, vw, 16);
            a0++;
        }
        vse32_v_f32m4(c0, vacc, vl);
        if(nc >= 4){
            c0 = (float*) ((uintptr_t) c0 + cn_stride);
            a0 = (const void*) ((uintptr_t) a0 - kc);
        }
        nc -= vl;
    } while (nc != 0);
}

void xnn_f32_gemm_ukernel_4x16__rvv_u4v(
        size_t mr,
        size_t nc,
        size_t kc,
        const float* restrict a,
        size_t a_stride,
        const float* restrict w,
        float* restrict c,
        size_t cm_stride,
        size_t cn_stride,
        const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(mr != 0);
    assert(mr <= 4); // max process 1 row
    assert(nc != 0);
    assert(kc != 0);
    assert(kc % sizeof(float) == 0);
    assert(a != NULL);
    assert(w != NULL);
    assert(c != NULL);

    const float* a0 = a;
    float* c0 = c;
    const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
    float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
    if XNN_UNPREDICTABLE(mr < 2) {
        a1 = a0;
        c1 = c0;
    }
    const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
    float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
    if XNN_UNPREDICTABLE(mr <= 2) {
        a2 = a1;
        c2 = c1;
    }
    const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
    float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
    if XNN_UNPREDICTABLE(mr != 4) {
        a3 = a2;
        c3 = c2;
    }

    size_t kcl = kc / sizeof(float);

    do {
        size_t vl = vsetvl_e32m4(nc); // vector length
        vfloat32m4_t vacc0 = vle32_v_f32m4(w, 16); // 1st row count
        vfloat32m4_t vacc1 = vle32_v_f32m4(w, 16); // 1st row count
        vfloat32m4_t vacc2 = vle32_v_f32m4(w, 16); // 1st row count
        vfloat32m4_t vacc3 = vle32_v_f32m4(w, 16); // 1st row count
        w += 16;
        for(size_t k = 0; k < kcl ; k++){
            vfloat32m4_t vw = vle32_v_f32m4(w, 16);
            w += 16;
            vacc0 = vfmacc_vf_f32m4(vacc0, *a0, vw, 16); // update 1st row count
            vacc1 = vfmacc_vf_f32m4(vacc1, *a1, vw, 16); // update 1st row count
            vacc2 = vfmacc_vf_f32m4(vacc2, *a2, vw, 16); // update 1st row count
            vacc3 = vfmacc_vf_f32m4(vacc3, *a3, vw, 16); // update 1st row count
            a0++;
            a1++;
            a2++;
            a3++;
        }
        vse32_v_f32m4(c0, vacc0, vl); // store 1st row result
        vse32_v_f32m4(c1, vacc1, vl); // store 1st row result
        vse32_v_f32m4(c2, vacc2, vl); // store 1st row result
        vse32_v_f32m4(c3, vacc3, vl); // store 1st row result
        if(nc >= 4){
            c0 = (float*) ((uintptr_t) c0 + cn_stride); // update 1st row matrix C pointer
            c1 = (float*) ((uintptr_t) c1 + cn_stride); // update 1st row matrix C pointer
            c2 = (float*) ((uintptr_t) c2 + cn_stride); // update 1st row matrix C pointer
            c3 = (float*) ((uintptr_t) c3 + cn_stride); // update 1st row matrix C pointer
            a0 = (const void*) ((uintptr_t) a0 - kc); // update 1st row matrix A pointer
            a1 = (const void*) ((uintptr_t) a1 - kc); // update 1st row matrix A pointer
            a2 = (const void*) ((uintptr_t) a2 - kc); // update 1st row matrix A pointer
            a3 = (const void*) ((uintptr_t) a3 - kc); // update 1st row matrix A pointer
        }
        nc -= vl;
    } while (nc != 0);
}

void xnn_f32_gemm_ukernel_1x32__rvv_u8v(
        size_t mr,
        size_t nc,
        size_t kc,
        const float* restrict a,
        size_t a_stride,
        const float* restrict w,
        float* restrict c,
        size_t cm_stride,
        size_t cn_stride,
        const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
     assert(mr != 0);
    assert(mr <= 1);
    assert(nc != 0);
    assert(kc != 0);
    assert(kc % sizeof(float) == 0);
    assert(a != NULL);
    assert(w != NULL);
    assert(c != NULL);

    const float* a0 = a;
    float* c0 = c;
    size_t kcl = kc / sizeof(float);

    do {
        size_t vl = vsetvl_e32m8(nc);
        vfloat32m8_t vacc = vle32_v_f32m8(w, 32);
        w += 32;
        for(size_t k = 0; k < kcl ; k++){
            vfloat32m8_t vw = vle32_v_f32m8(w, 32);
            w += 32;
            vacc = vfmacc_vf_f32m8(vacc, *a0, vw, 32);
            a0++;
        }
        vse32_v_f32m8(c0, vacc, vl);
        if(nc >= 4){
            c0 = (float*) ((uintptr_t) c0 + cn_stride);
            a0 = (const void*) ((uintptr_t) a0 - kc);
        }
        nc -= vl;
    } while (nc != 0);
}

void xnn_f32_gemm_ukernel_4x32__rvv_u8v(
    size_t mr,
        size_t nc,
        size_t kc,
        const float* restrict a,
        size_t a_stride,
        const float* restrict w,
        float* restrict c,
        size_t cm_stride,
        size_t cn_stride,
        const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    const float* a0 = a;
    float* c0 = c;
    const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
    float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
    if XNN_UNPREDICTABLE(mr < 2) {
        a1 = a0;
        c1 = c0;
    }
    const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
    float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
    if XNN_UNPREDICTABLE(mr <= 2) {
        a2 = a1;
        c2 = c1;
    }
    const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
    float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
    if XNN_UNPREDICTABLE(mr != 4) {
        a3 = a2;
        c3 = c2;
    }

    size_t kcl = kc / sizeof(float);

    do
    {
        size_t vl = vsetvl_e32m8(nc);
        vfloat32m8_t vacc0 = vle32_v_f32m8(w, 32); // 1st row count
        vfloat32m8_t vacc1 = vle32_v_f32m8(w, 32); // 1st row count
        vfloat32m8_t vacc2 = vle32_v_f32m8(w, 32); // 1st row count
        vfloat32m8_t vacc3 = vle32_v_f32m8(w, 32); // 1st row count
        w += 32;
        for(size_t k = 0; k < kcl ; k++){
            vfloat32m8_t vw = vle32_v_f32m8(w, 32);
            w += 32;
            vacc0 = vfmacc_vf_f32m8(vacc0, *a0, vw, 32); // update 1st row count
            vacc1 = vfmacc_vf_f32m8(vacc1, *a1, vw, 32); // update 1st row count
            vacc2 = vfmacc_vf_f32m8(vacc2, *a2, vw, 32); // update 1st row count
            vacc3 = vfmacc_vf_f32m8(vacc3, *a3, vw, 32); // update 1st row count
            a0++;
            a1++;
            a2++;
            a3++;
        }
        vse32_v_f32m8(c0, vacc0, vl); // store 1st row result
        vse32_v_f32m8(c1, vacc1, vl); // store 1st row result
        vse32_v_f32m8(c2, vacc2, vl); // store 1st row result
        vse32_v_f32m8(c3, vacc3, vl); // store 1st row result
        if(nc >= 4){
            c0 = (float*) ((uintptr_t) c0 + cn_stride); // update 1st row matrix C pointer
            c1 = (float*) ((uintptr_t) c1 + cn_stride); // update 1st row matrix C pointer
            c2 = (float*) ((uintptr_t) c2 + cn_stride); // update 1st row matrix C pointer
            c3 = (float*) ((uintptr_t) c3 + cn_stride); // update 1st row matrix C pointer
            a0 = (const void*) ((uintptr_t) a0 - kc); // update 1st row matrix A pointer
            a1 = (const void*) ((uintptr_t) a1 - kc); // update 1st row matrix A pointer
            a2 = (const void*) ((uintptr_t) a2 - kc); // update 1st row matrix A pointer
            a3 = (const void*) ((uintptr_t) a3 - kc); // update 1st row matrix A pointer
        }
        nc -= vl;
    } while (nc != 0);
    
}


//void xnn_f32_gemm_ukernel_4x2__rvv_u1v(
//        size_t mr,
//        size_t nc,
//        size_t kc,
//        const float* restrict a,
//        size_t a_stride,
//        const float* restrict w,
//        float* restrict c,
//        size_t cm_stride,
//        size_t cn_stride,
//        const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
//{
//    assert(mr != 0);
//    assert(mr <= 4); // max process 4 row
//    assert(nc != 0);
//    assert(kc != 0);
//    assert(kc % sizeof(float) == 0);
//    assert(a != NULL);
//    assert(w != NULL);
//    assert(c != NULL);
//
//    // each row
//    for (size_t m = 0; m < mr; ++m) {
//        const float* a0 = a + m * a_stride; // support multi-row
//        float* c0 = c + m * cm_stride; // support multi-row
//
//        size_t kcl = kc / sizeof(float);
//        size_t vl = vsetvl_e32m1(2); // set the vector length to 2 for 2 cols processing
//        vfloat32m1_t vacc = vfsub_vv_f32m1(vle32_v_f32m1(c0, vl), vle32_v_f32m1(c0, vl), vl); // initialize vacc to 0
//
//        for(size_t k = 0; k < kcl; ++k) {
//            vfloat32m1_t vw = vle32_v_f32m1(w, vl);
//            w += vl;
//            vacc = vfmacc_vf_f32m1(vacc, a0[k], vw, vl); // correct multiplication and accumulation
//        }
//
//        vse32_v_f32m1(c0, vacc, vl); // store result
//    }
//}

//void xnn_f32_gemm_relu_ukernel_1x4__rvv_u1v(
//        size_t mr,
//        size_t nc,
//        size_t kc,
//        const float* restrict a,
//        size_t a_stride,
//        const float* restrict w,
//        float* restrict c,
//        size_t cm_stride,
//        size_t cn_stride,
//        const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
//{
//    assert(mr != 0);
//    assert(mr <= 1);
//    assert(nc != 0);
//    assert(kc != 0);
//    assert(kc % sizeof(float) == 0);
//    assert(a != NULL);
//    assert(w != NULL);
//    assert(c != NULL);
//
//    const float* a0 = a;
//    float* c0 = c;
//    size_t kcl = kc / sizeof(float);
//
//    do {
//        size_t vl = vsetvl_e32m1(nc);
//        vfloat32m1_t vacc = vle32_v_f32m1(w, vl);
//        w += vl;
//        for(size_t k = 0; k < kcl ; k++){
//            vfloat32m1_t vw = vle32_v_f32m1(w, vl);
//            w += vl;
//            vacc = vfmacc_vf_f32m1(vacc, *a0, vw, vl);
//            a0++;
//        }
//        vacc = vfmax_vf_f32m1(vacc, 0.0, vl);
//
//        vse32_v_f32m1(c0, vacc, vl);
//        if(nc >= 4){
//            c0 = (float*) ((uintptr_t) c0 + cn_stride);
//            a0 = (const void*) ((uintptr_t) a0 - kc);
//        }
//        nc -= vl;
//    } while (nc != 0);
//}

//void xnn_f32_gemm_relu_ukernel_2x4__rvv_u1v(
//        size_t mr,
//        size_t nc,
//        size_t kc,
//        const float* restrict a,
//        size_t a_stride,
//        const float* restrict w,
//        float* restrict c,
//        size_t cm_stride,
//        size_t cn_stride,
//        const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
//{
//    assert(mr != 0);
//    assert(mr <= 2); // max process 2 row
//    assert(nc != 0);
//    assert(kc != 0);
//    assert(kc % sizeof(float) == 0);
//    assert(a != NULL);
//    assert(w != NULL);
//    assert(c != NULL);
//
//    const float* a0 = a;
//    const float* a1 = a + a_stride;
//    float* c0 = c;
//    float* c1 = c + cm_stride;
//    size_t kcl = kc / sizeof(float);
//
//    do {
//        size_t vl = vsetvl_e32m1(nc);
//        vfloat32m1_t vacc0 = vfsub_vv_f32m1(vle32_v_f32m1(c0, vl), vle32_v_f32m1(c0, vl), vl); // 0th row count
//        vfloat32m1_t vacc1 = vfsub_vv_f32m1(vle32_v_f32m1(c1, vl), vle32_v_f32m1(c1, vl), vl); // 1st row count
//        w += vl;
//        for(size_t k = 0; k < kcl ; k++){
//            vfloat32m1_t va0 = vfmv_v_f_f32m1(*a0, vl);
//            vfloat32m1_t va1 = vfmv_v_f_f32m1(*a1, vl);
//            vfloat32m1_t vw = vle32_v_f32m1(w, vl);
//            vacc0 = vfmacc_vv_f32m1(vacc0, va0, vw, vl);
//            vacc1 = vfmacc_vv_f32m1(vacc1, va1, vw, vl);
//            a0++;
//            a1++;
//        }
//
//        vacc0 = vfmax_vf_f32m1(vacc0, 0.0, vl);
//        vacc1 = vfmax_vf_f32m1(vacc1, 0.0, vl);
//        vse32_v_f32m1(c0, vacc0, vl);
//        vse32_v_f32m1(c1, vacc1, vl);
//
//        if(nc >= 4){
//            c0 = (float*) ((uintptr_t) c0 + cn_stride);
//            c1 = (float*) ((uintptr_t) c1 + cn_stride);
//
//            a0 = (const void*) ((uintptr_t) a0 - kc);
//            a1 = (const void*) ((uintptr_t) a1 - kc);
//        }
//        nc -= vl;
//    } while (nc != 0);
//}

void xnn_x32_packw_gemm_goi_ukernel_x4__rvv_float_u4(
        size_t g,
        size_t nc,
        size_t kc,
        size_t nr,
        size_t kr,
        size_t sr,
        const uint32_t* weights,
        const uint32_t* bias,
        const void* scale,
        uint32_t* packed_weights,
        size_t extra_bytes,
        const void* params)
{
    assert(g != 0);
    assert(nc != 0);
    assert(kc != 0);
    assert(nr == 4);
    assert(kr == 1);
    assert(sr == 1);
    assert(weights != NULL);
    assert(packed_weights != NULL);

    float* out = (float*) packed_weights;
    const float* b = (const float*) bias;

    do {
        // NC main loop multiple of 4
        const float* w0 = (const float*) weights;
        size_t n = nc;

        do {
            size_t vl = vsetvl_e32m1(n);
            vfloat32m1_t vacc;
            if XNN_LIKELY(b != NULL) {
                vacc = vle32_v_f32m1(b, vl);
                b += 4;
            } else {
                vacc = vfmv_v_f_f32m1(0.0f, vl);
            }
            vse32_v_f32m1(out, vacc, vl);
            out += 4;

            size_t k = kc;
/*            for (; k >= 8; k -= 8) {
                vfloat32m1_t vacc0 = vlse32_v_f32m1(w0, kc * sizeof(float), vl); //按列取vl个数据存入寄存器，行长为kc
                vfloat32m1_t vacc1 = vlse32_v_f32m1(w0 + 1, kc * sizeof(float), vl); //按列取vl个数据存入寄存器，行长为kc
                vfloat32m1_t vacc2 = vlse32_v_f32m1(w0 + 2, kc * sizeof(float), vl); //按列取vl个数据存入寄存器，行长为kc
                vfloat32m1_t vacc3 = vlse32_v_f32m1(w0 + 3, kc * sizeof(float), vl); //按列取vl个数据存入寄存器，行长为kc
                vfloat32m1_t vacc4 = vlse32_v_f32m1(w0 + 4, kc * sizeof(float), vl); //按列取vl个数据存入寄存器，行长为kc
                vfloat32m1_t vacc5 = vlse32_v_f32m1(w0 + 5, kc * sizeof(float), vl); //按列取vl个数据存入寄存器，行长为kc
                vfloat32m1_t vacc6 = vlse32_v_f32m1(w0 + 6, kc * sizeof(float), vl); //按列取vl个数据存入寄存器，行长为kc
                vfloat32m1_t vacc7 = vlse32_v_f32m1(w0 + 7, kc * sizeof(float), vl); //按列取vl个数据存入寄存器，行长为kc
                vse32_v_f32m1(out, vacc0, vl);
                vse32_v_f32m1(out + 4, vacc1, vl);
                vse32_v_f32m1(out + 8, vacc2, vl);
                vse32_v_f32m1(out + 12, vacc3, vl);
                vse32_v_f32m1(out + 16, vacc0, vl);
                vse32_v_f32m1(out + 20, vacc1, vl);
                vse32_v_f32m1(out + 24, vacc2, vl);
                vse32_v_f32m1(out + 28, vacc3, vl);
                w0 += 8;
                if XNN_UNPREDICTABLE(vl < 2) {
                    out[1] = out[0];
                    out[5] = out[4];
                    out[9] = out[8];
                    out[13] = out[12];
                    out[17] = out[16];
                    out[21] = out[20];
                    out[25] = out[24];
                    out[29] = out[28];
                }
                if XNN_UNPREDICTABLE(vl <= 2) {
                    out[2] = out[1];
                    out[6] = out[5];
                    out[10] = out[9];
                    out[14] = out[13];
                    out[18] = out[17];
                    out[22] = out[21];
                    out[26] = out[25];
                    out[30] = out[29];
                }
                out += 32;
            }*/
            for (; k >= 4; k -= 4) {
                vfloat32m1_t vacc0 = vlse32_v_f32m1(w0, kc * sizeof(float), vl); //按列取vl个数据存入寄存器，行长为kc
                vfloat32m1_t vacc1 = vlse32_v_f32m1(w0 + 1, kc * sizeof(float), vl); //按列取vl个数据存入寄存器，行长为kc
                vfloat32m1_t vacc2 = vlse32_v_f32m1(w0 + 2, kc * sizeof(float), vl); //按列取vl个数据存入寄存器，行长为kc
                vfloat32m1_t vacc3 = vlse32_v_f32m1(w0 + 3, kc * sizeof(float), vl); //按列取vl个数据存入寄存器，行长为kc
                vse32_v_f32m1(out, vacc0, vl);
                vse32_v_f32m1(out + 4, vacc1, vl);
                vse32_v_f32m1(out + 8, vacc2, vl);
                vse32_v_f32m1(out + 12, vacc3, vl);
                w0 += 4;
                if XNN_UNPREDICTABLE(vl < 2) {
                    out[1] = out[0];
                    out[5] = out[4];
                    out[9] = out[8];
                    out[13] = out[12];
                }
                if XNN_UNPREDICTABLE(vl <= 2) {
                    out[2] = out[1];
                    out[6] = out[5];
                    out[10] = out[9];
                    out[14] = out[13];
                }
                out += 16;
            }
            for (; k != 0; --k) {
                vacc = vlse32_v_f32m1(w0, kc * sizeof(float), vl); //按列取vl个数据存入寄存器，行长为kc
                vse32_v_f32m1(out, vacc, vl);
                w0++;
                if XNN_UNPREDICTABLE(vl < 2) {
                    out[1] = out[0];
                }
                if XNN_UNPREDICTABLE(vl <= 2) {
                    out[2] = out[1];
                }
                out += 4;
            }

            out = (float*) ((uintptr_t) out + extra_bytes);
            w0 += kc * 3;
            n -= vl;
        } while (n != 0);

        weights += nc * kc;

    } while (--g != 0);
}



//void xnn_x32_packw_gemm_goi_ukernel_x8__rvv_float_u4(
//        size_t g,
//        size_t nc,
//        size_t kc,
//        size_t nr,
//        size_t kr,
//        size_t sr,
//        const uint32_t* weights,
//        const uint32_t* bias,
//        const void* scale,
//        uint32_t* packed_weights,
//        size_t extra_bytes,
//        const void* params)
//{
//    assert(g != 0);
//    assert(nc != 0);
//    assert(kc != 0);
//    assert(nr == 8);
//    assert(kr == 1);
//    assert(sr == 1);
//    assert(weights != NULL);
//    assert(packed_weights != NULL);
//
//    float* out = (float*) packed_weights;
//    const float* b = (const float*) bias;
//
//    do {
//        // NC main loop multiple of 8
//        const float* w0 = (const float*) weights;
//        size_t n = nc;
//
//        do {
//            size_t vl = vsetvl_e32m2(n);
//            vfloat32m2_t vacc;
//            if XNN_LIKELY(b != NULL) {
//                vacc = vle32_v_f32m2(b, vl);
//                b += 8;
//            } else {
//                vacc = vfmv_v_f_f32m2(0.0f, vl);
//            }
//            vse32_v_f32m2(out, vacc, vl);
//            out += 8;
//
//            size_t k = kc;
///*            for (; k >= 8; k -= 8) {
//                vfloat32m1_t vacc0 = vlse32_v_f32m1(w0, kc * sizeof(float), vl); //按列取vl个数据存入寄存器，行长为kc
//                vfloat32m1_t vacc1 = vlse32_v_f32m1(w0 + 1, kc * sizeof(float), vl); //按列取vl个数据存入寄存器，行长为kc
//                vfloat32m1_t vacc2 = vlse32_v_f32m1(w0 + 2, kc * sizeof(float), vl); //按列取vl个数据存入寄存器，行长为kc
//                vfloat32m1_t vacc3 = vlse32_v_f32m1(w0 + 3, kc * sizeof(float), vl); //按列取vl个数据存入寄存器，行长为kc
//                vfloat32m1_t vacc4 = vlse32_v_f32m1(w0 + 4, kc * sizeof(float), vl); //按列取vl个数据存入寄存器，行长为kc
//                vfloat32m1_t vacc5 = vlse32_v_f32m1(w0 + 5, kc * sizeof(float), vl); //按列取vl个数据存入寄存器，行长为kc
//                vfloat32m1_t vacc6 = vlse32_v_f32m1(w0 + 6, kc * sizeof(float), vl); //按列取vl个数据存入寄存器，行长为kc
//                vfloat32m1_t vacc7 = vlse32_v_f32m1(w0 + 7, kc * sizeof(float), vl); //按列取vl个数据存入寄存器，行长为kc
//                vse32_v_f32m1(out, vacc0, vl);
//                vse32_v_f32m1(out + 4, vacc1, vl);
//                vse32_v_f32m1(out + 8, vacc2, vl);
//                vse32_v_f32m1(out + 12, vacc3, vl);
//                vse32_v_f32m1(out + 16, vacc0, vl);
//                vse32_v_f32m1(out + 20, vacc1, vl);
//                vse32_v_f32m1(out + 24, vacc2, vl);
//                vse32_v_f32m1(out + 28, vacc3, vl);
//                w0 += 8;
//                if XNN_UNPREDICTABLE(vl < 2) {
//                    out[1] = out[0];
//                    out[5] = out[4];
//                    out[9] = out[8];
//                    out[13] = out[12];
//                    out[17] = out[16];
//                    out[21] = out[20];
//                    out[25] = out[24];
//                    out[29] = out[28];
//                }
//                if XNN_UNPREDICTABLE(vl <= 2) {
//                    out[2] = out[1];
//                    out[6] = out[5];
//                    out[10] = out[9];
//                    out[14] = out[13];
//                    out[18] = out[17];
//                    out[22] = out[21];
//                    out[26] = out[25];
//                    out[30] = out[29];
//                }
//                out += 32;
//            }*/
//            for (; k >= 4; k -= 4) {
//                vfloat32m1_t vacc0 = vlse32_v_f32m1(w0, kc * sizeof(float), vl); //按列取vl个数据存入寄存器，行长为kc
//                vfloat32m1_t vacc1 = vlse32_v_f32m1(w0 + 1, kc * sizeof(float), vl); //按列取vl个数据存入寄存器，行长为kc
//                vfloat32m1_t vacc2 = vlse32_v_f32m1(w0 + 2, kc * sizeof(float), vl); //按列取vl个数据存入寄存器，行长为kc
//                vfloat32m1_t vacc3 = vlse32_v_f32m1(w0 + 3, kc * sizeof(float), vl); //按列取vl个数据存入寄存器，行长为kc
//                vse32_v_f32m1(out, vacc0, vl);
//                vse32_v_f32m1(out + 4, vacc1, vl);
//                vse32_v_f32m1(out + 8, vacc2, vl);
//                vse32_v_f32m1(out + 12, vacc3, vl);
//                w0 += 4;
//                if XNN_UNPREDICTABLE(vl < 2) {
//                    out[1] = out[0];
//                    out[5] = out[4];
//                    out[9] = out[8];
//                    out[13] = out[12];
//                }
//                if XNN_UNPREDICTABLE(vl <= 2) {
//                    out[2] = out[1];
//                    out[6] = out[5];
//                    out[10] = out[9];
//                    out[14] = out[13];
//                }
//                out += 16;
//            }
//            for (; k != 0; --k) {
//                vacc = vlse32_v_f32m1(w0, kc * sizeof(float), vl); //按列取vl个数据存入寄存器，行长为kc
//                vse32_v_f32m1(out, vacc, vl);
//                w0++;
//                if XNN_UNPREDICTABLE(vl < 2) {
//                    out[1] = out[0];
//                }
//                if XNN_UNPREDICTABLE(vl <= 2) {
//                    out[2] = out[1];
//                }
//                out += 4;
//            }
//
//            out = (float*) ((uintptr_t) out + extra_bytes);
//            w0 += kc * 3;
//            n -= vl;
//        } while (n != 0);
//
//        weights += nc * kc;
//
//    } while (--g != 0);
//}

extern XNN_INTERNAL const uint32_t xnn_table_exp2minus_k_over_64[64];
// void xnn_f32_vsigmoid_ukernel__rvv_u2v(
//         size_t batch,
//         const float* input,
//         float* output,
//         const union xnn_f32_sigmoid_params params[restrict XNN_MIN_ELEMENTS(1)])
// {
//     assert(batch != 0);
//     assert(batch % sizeof(float) == 0);
//     assert(input != NULL);
//     assert(output != NULL);

//     const float vmagic_bias = params->scalar_rr2_lut64_p2.magic_bias;
//     const float vminus_log2e = params->scalar_rr2_lut64_p2.minus_log2e;
//     const uint32_t vindex_mask = UINT32_C(0x3F);
//     const float vln2_hi = params->scalar_rr2_lut64_p2.ln2_hi;
//     const float vln2_lo = params->scalar_rr2_lut64_p2.ln2_lo;
//     const float vc2 = params->scalar_rr2_lut64_p2.c2;
//     const float vone = params->scalar_rr2_lut64_p2.one;
//     const float vdenorm_cutoff = params->scalar_rr2_lut64_p2.denorm_cutoff;

//     size_t size = batch / sizeof(float);
//     do {
//         const size_t vl = vsetvl_e32m2(size);
//         vfloat32m2_t vx = vle32_v_f32m2(input, vl);
//         input += vl;
//         // get abs
//         vfloat32m2_t vz = vfabs_v_f32m2(vx, vl);
//         // vz*(-log2(e))+magic_bias
//         vfloat32m2_t vn = vfadd_vf_f32m2(vfmul_vf_f32m2(vz, vminus_log2e, vl), vmagic_bias, vl);
//         // get exponent
//         vuint32m2_t ve = vsll_vx_u32m2(vreinterpret_v_f32m2_u32m2(vn), 17, vl);
//         // find index in lookup table using mask
//         vuint32m2_t vidx = vand_vx_u32m2(vreinterpret_v_f32m2_u32m2(vn), vindex_mask, vl);
//         vfloat32m2_t vs = vreinterpret_v_u32m2_f32m2(vadd_vv_u32m2(vloxei32_v_u32m2(xnn_table_exp2minus_k_over_64, vmul_vx_u32m2(vidx, 4, vl), vl), ve, vl));
//         // remove magic bias
//         vn = vfsub_vf_f32m2(vn, vmagic_bias, vl);
//         // find logarithm
//         vfloat32m2_t vt = vfadd_vv_f32m2(vfmul_vf_f32m2(vn, vln2_hi, vl), vz, vl);
//         vt = vfmacc_vf_f32m2(vt, vln2_lo, vn, vl);
//         // calculate the quadratic term logarithmically.
//         vfloat32m2_t vp = vfmul_vf_f32m2(vt, vc2, vl);
//         vp = vfsub_vv_f32m2(vt, vfmul_vv_f32m2(vp, vt, vl), vl);
//         // caculate sigmoid polynomial approximation
//         vfloat32m2_t vy = vfsub_vv_f32m2(vs, vfmul_vv_f32m2(vs, vp, vl), vl);
//         vfloat32m2_t vd = vfadd_vf_f32m2(vy, vone, vl);
//         vfloat32m2_t vf = vfdiv_vv_f32m2(vy, vd, vl);

//         vbool16_t mask = vmfgt_vf_f32m2_b16 (vz, vdenorm_cutoff, vl);
//         vf = vfmerge_vfm_f32m2(mask, vf, 0.0f, vl);

//         mask = vmfgt_vf_f32m2_b16 (vx, 0.0f, vl);
//         vf = vfmul_vf_f32m2_m(mask, vf, vf, -1.0f, vl);
//         vf = vfadd_vf_f32m2_m(mask, vf, vf, vone, vl);

//         // store result
//         vse32_v_f32m2(output, vf, vl);

//         output += vl;
//         size -= vl;
//     } while (size > 0);
// }

void xnn_f32_vsigmoid_ukernel__thead_u2v(
        size_t batch,
        const float* input,
        float* output,
        const union xnn_f32_sigmoid_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(batch != 0);
    assert(batch % sizeof(float) == 0);
    assert(input != NULL);
    assert(output != NULL);

    float *input_data = input;
    float *output_data = output;

    size_t size = batch / sizeof(float);
    while (size > 0) {
        size_t vl = vsetvl_e32m2(size);

        vfloat32m2_t _val = vle32_v_f32m2(input_data, vl);  // val
        _val = vfmul_vf_f32m2(_val, -1.0f, vl);
        vfloat32m2_t _output_data = exp_ps_vfloat32m2(_val, vl);
        _output_data = vfadd_vf_f32m2(_output_data, 1.0f, vl);
        _output_data = vfrdiv_vf_f32m2(_output_data, 1.0f, vl);
        vse32_v_f32m2(output_data, _output_data, vl);

        input_data += vl;
        output_data += vl;
        size -= vl;
    }
}

void xnn_f32_vsigmoid_ukernel__thead_u1v(
        size_t batch,
        const float* input,
        float* output,
        const union xnn_f32_sigmoid_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(batch != 0);
    assert(batch % sizeof(float) == 0);
    assert(input != NULL);
    assert(output != NULL);

    float *input_data = input;
    float *output_data = output;

    size_t size = batch / sizeof(float);
    while (size > 0) {
        size_t vl = vsetvl_e32m1(size);

        vfloat32m1_t _val = vle32_v_f32m1(input_data, vl);  // val
        _val = vfmul_vf_f32m1(_val, -1.0f, vl);
        vfloat32m1_t _output_data = exp_ps_vfloat32m1(_val, vl);
        _output_data = vfadd_vf_f32m1(_output_data, 1.0f, vl);
        _output_data = vfrdiv_vf_f32m1(_output_data, 1.0f, vl);
        vse32_v_f32m1(output_data, _output_data, vl);

        input_data += vl;
        output_data += vl;
        size -= vl;
    }
}

void xnn_f32_vsigmoid_ukernel__thead_u4v(
        size_t batch,
        const float* input,
        float* output,
        const union xnn_f32_sigmoid_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(batch != 0);
    assert(batch % sizeof(float) == 0);
    assert(input != NULL);
    assert(output != NULL);

    float *input_data = input;
    float *output_data = output;

    size_t size = batch / sizeof(float);
    while (size > 0) {
        size_t vl = vsetvl_e32m4(size);

        vfloat32m4_t _val = vle32_v_f32m4(input_data, vl);  // val
        _val = vfmul_vf_f32m4(_val, -1.0f, vl);
        vfloat32m4_t _output_data = exp_ps_vfloat32m4(_val, vl);
        _output_data = vfadd_vf_f32m4(_output_data, 1.0f, vl);
        _output_data = vfrdiv_vf_f32m4(_output_data, 1.0f, vl);
        vse32_v_f32m4(output_data, _output_data, vl);

        input_data += vl;
        output_data += vl;
        size -= vl;
    }
}

void xnn_f32_vsigmoid_ukernel__thead_u8v(
        size_t batch,
        const float* input,
        float* output,
        const union xnn_f32_sigmoid_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(batch != 0);
    assert(batch % sizeof(float) == 0);
    assert(input != NULL);
    assert(output != NULL);

    float *input_data = input;
    float *output_data = output;

    size_t size = batch / sizeof(float);
    while (size > 0) {
        size_t vl = vsetvl_e32m8(size);

        vfloat32m8_t _val = vle32_v_f32m8(input_data, vl);  // val
        _val = vfmul_vf_f32m8(_val, -1.0f, vl);
        vfloat32m8_t _output_data = exp_ps_vfloat32m8(_val, vl);
        _output_data = vfadd_vf_f32m8(_output_data, 1.0f, vl);
        _output_data = vfrdiv_vf_f32m8(_output_data, 1.0f, vl);
        vse32_v_f32m8(output_data, _output_data, vl);

        input_data += vl;
        output_data += vl;
        size -= vl;
    }
}

//void xnn_f32_prelu_ukernel__rvv_2x8(
//        size_t rows,
//        size_t channels,
//        const float* restrict input,
//        size_t input_stride,
//        const float* restrict weights,
//        float* restrict output,
//        size_t output_stride)
//{
//    assert(rows != 0);
//    assert(channels != 0);
//    assert(channels % sizeof(float) == 0);
//
//    const float* i0 = input;
//    float* o0 = output;
//    const float* i1 = (const float*) ((uintptr_t) i0 + input_stride);
//    float* o1 = (float*) ((uintptr_t) o0 + output_stride);
//
//    const size_t input_increment = input_stride * 2 - channels;
//    const size_t output_increment = output_stride * 2 - channels;
//
//    do {
//        if XNN_UNPREDICTABLE(rows < 2) { // if rows < 2, process 1 row
//            i1 = i0;
//            o1 = o0;
//        }
//
//        const float* w = weights; // pointer to first element of weights
//        size_t c = channels; // initialize number of channels
//        for(; c >= 8 * sizeof(float); c -= 8 * sizeof(float)) {
//            const size_t vl = vsetvl_e32m1(c); // set vector length
//            const vfloat32m1_t vw0123 = vle32_v_f32m1(w, vl); // load 4 weights
//            w += 4;
//            const vfloat32m1_t vw4567 = vle32_v_f32m1(w, vl); // load 4 weights
//            w += 4;
//
//            const vfloat32m1_t vi0x0123 = vle32_v_f32m1(i0, vl); // load 4 input
//            i0 += 4;
//            const vfloat32m1_t vi0x4567 = vle32_v_f32m1(i0, vl); // load 4 input
//            i0 += 4;
//            const vfloat32m1_t vi1x0123 = vle32_v_f32m1(i1, vl); // load 4 input
//            i1 += 4;
//            const vfloat32m1_t vi1x4567 = vle32_v_f32m1(i1, vl); // load 4 input
//            i1 += 4;
//
//            vfloat32m1_t vacc0x0123 = vfmul_vv_f32m1(vi0x0123, vw0123, vl); // multiplication
//            //neon: const uint32x4_t vm0x0123 = vcltq_s32(vreinterpretq_s32_f32(vi0x0123), vmovq_n_s32(0));
//            const vbool32_t vm0x0123 = vmflt_vf_f32m1_b32(vi0x0123, .0f, vl);
//            vfloat32m1_t vacc0x4567 = vfmul_vv_f32m1(vi0x4567, vw4567, vl); // multiplication
//            const vbool32_t vm0x4567 = vmflt_vf_f32m1_b32(vi0x4567, .0f, vl);
//            vfloat32m1_t vacc1x0123 = vfmul_vv_f32m1(vi1x0123, vw0123, vl); // multiplication
//            const vbool32_t vm1x0123 = vmflt_vf_f32m1_b32(vi1x0123, .0f, vl);
//            vfloat32m1_t vacc1x4567 = vfmul_vv_f32m1(vi1x4567, vw4567, vl); // multiplication
//            const vbool32_t vm1x4567 = vmflt_vf_f32m1_b32(vi1x4567, .0f, vl);
//            // neon:
//            // vacc0x0123 = vbslq_f32(vm0x0123, vacc0x0123, vi0x0123);
//            // vacc0x4567 = vbslq_f32(vm0x4567, vacc0x4567, vi0x4567);
//            // vacc1x0123 = vbslq_f32(vm1x0123, vacc1x0123, vi1x0123);
//            // vacc1x4567 = vbslq_f32(vm1x4567, vacc1x4567, vi1x4567);
//            vacc0x0123 = vmerge_vvm_f32m1(vm0x0123, vacc0x0123, vi0x0123, vl);
//            vacc0x4567 = vmerge_vvm_f32m1(vm0x4567, vacc0x4567, vi0x4567, vl);
//            vacc1x0123 = vmerge_vvm_f32m1(vm1x0123, vacc1x0123, vi1x0123, vl);
//            vacc1x4567 = vmerge_vvm_f32m1(vm1x4567, vacc1x4567, vi1x4567, vl);
//
//            vse32_v_f32m1(o0, vacc0x0123, vl); // store result
//            o0 += 4;
//            vse32_v_f32m1(o0, vacc0x4567, vl); // store result
//            o0 += 4;
//            vse32_v_f32m1(o1, vacc1x0123, vl); // store result
//            o1 += 4;
//            vse32_v_f32m1(o1, vacc1x4567, vl); // store result
//            o1 += 4;
//        }
//
//        for (; c >= 4 * sizeof(float); c -= 4 * sizeof(float)) { // process 4 cols
//            const size_t vl = vsetvl_e32m1(c);
//            const vfloat32m1_t vw0123 = vle32_v_f32m1(w, vl); // load 4 weights
//            w += 4;
//            const vfloat32m1_t vi0x0123 = vle32_v_f32m1(i0, vl); // load 4 input
//            i0 += 4;
//            const vfloat32m1_t vi1x0123 = vle32_v_f32m1(i1, vl); // load 4 input
//            i1 += 4;
//
//            vfloat32m1_t vacc0x0123 = vfmul_vv_f32m1(vi0x0123, vw0123, vl); // multiplication
//            const vbool32_t vm0x0123 = vmflt_vf_f32m1_b32(vi0x0123, .0f, vl);
//            vfloat32m1_t vacc1x0123 = vfmul_vv_f32m1(vi1x0123, vw0123, vl); // multiplication
//            const vbool32_t vm1x0123 = vmflt_vf_f32m1_b32(vi1x0123, .0f, vl);
//
//            vacc0x0123 = vmerge_vvm_f32m1(vm0x0123, vacc0x0123, vi0x0123, vl);
//            vacc1x0123 = vmerge_vvm_f32m1(vm1x0123, vacc1x0123, vi1x0123, vl);
//
//            vse32_v_f32m1(o0, vacc0x0123, vl); // store result
//            o0 += 4;
//            vse32_v_f32m1(o1, vacc1x0123, vl); // store result
//            o1 += 4;
//        }
//        if XNN_UNLIKELY(c != 0) { //
//            const size_t vl = vsetvl_e32m1(c);
//            const vfloat32m1_t vw0123 = vle32_v_f32m1(w, vl); // load 4 weights
//            w += 4;
//            const vfloat32m1_t vi0x0123 = vle32_v_f32m1(i0, vl); // load 4 input
//            i0 = (const float*) ((uintptr_t) i0 + c);
//            const vfloat32m1_t vi1x0123 = vle32_v_f32m1(i1, vl); // load 4 input
//            i1 = (const float*) ((uintptr_t) i1 + c);
//
//            vfloat32m1_t vacc0x0123 = vfmul_vv_f32m1(vi0x0123, vw0123, vl); // multiplication
//            const vbool32_t vm0x0123 = vmflt_vf_f32m1_b32(vi0x0123, .0f, vl);
//            vfloat32m1_t vacc1x0123 = vfmul_vv_f32m1(vi1x0123, vw0123, vl); // multiplication
//            const vbool32_t vm1x0123 = vmflt_vf_f32m1_b32(vi1x0123, .0f, vl);
//
//            vacc0x0123 = vmerge_vvm_f32m1(vm0x0123, vacc0x0123, vi0x0123, vl);
//            vacc1x0123 = vmerge_vvm_f32m1(vm1x0123, vacc1x0123, vi1x0123, vl);
//
//            vse32_v_f32m1(o0, vacc0x0123, vl); // store result
//            o0 = (float*) ((uintptr_t) o0 + c);
//            vse32_v_f32m1(o1, vacc1x0123, vl); // store result
//            o1 = (float*) ((uintptr_t) o1 + c);
//        }
//        i0 = (const float*) ((uintptr_t) i0 + input_increment);
//        o0 = (float*) ((uintptr_t) o0 + output_increment);
//        i1 = (const float*) ((uintptr_t) i1 + input_increment);
//        o1 = (float*) ((uintptr_t) o1 + output_increment);
//        rows = doz(rows, 2);
//    } while (rows != 0);
//}

//向量除法
void xnn_f32_vdiv_minmax_ukernel__rvv_u2v(
        size_t batch,
        const float* input_a,
        const float* input_b,
        float* output,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(batch != 0);
    assert(batch % sizeof(float) == 0);
    assert(input_a != NULL);
    assert(input_b != NULL);
    assert(output != NULL);

    const float voutput_min = params->scalar.min;
    const float voutput_max = params->scalar.max;

    size_t size = batch / sizeof(float);
    do {
        // 动态设置向量长度
        const size_t vl = vsetvl_e32m2(size);

        // 加载输入向量
        vfloat32m2_t va = vle32_v_f32m2(input_a, vl);
        vfloat32m2_t vb = vle32_v_f32m2(input_b, vl);
        input_a += vl;
        input_b += vl;

        // 执行向量除法
        vfloat32m2_t vacc = vfdiv_vv_f32m2(va, vb, vl);

        // 应用最小值约束
        vacc = vfmax_vf_f32m2(vacc, voutput_min, vl);

        // 应用最大值约束
        vacc = vfmin_vf_f32m2(vacc, voutput_max, vl);

        // 存储结果
        vse32_v_f32m2(output, vacc, vl);
        output += vl;
        size -= vl;
    } while (size > 0);
}

//向量加法
void xnn_f32_vadd_minmax_ukernel__rvv_u2v(
        size_t batch,
        const float* input_a,
        const float* input_b,
        float* output,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(batch != 0);
    assert(batch % sizeof(float) == 0);
    assert(input_a != NULL);
    assert(input_b != NULL);
    assert(output != NULL);

    const float voutput_min = params->scalar.min;
    const float voutput_max = params->scalar.max;

    size_t size = batch / sizeof(float);
    do {
        // 动态设置向量长度
        const size_t vl = vsetvl_e32m2(size);

        // 加载输入向量
        vfloat32m2_t va = vle32_v_f32m2(input_a, vl);
        vfloat32m2_t vb = vle32_v_f32m2(input_b, vl);
        input_a += vl;
        input_b += vl;

        // 执行向量加法
        vfloat32m2_t vacc = vfadd_vv_f32m2(va, vb, vl);

        // 应用最小值约束
        vacc = vfmax_vf_f32m2(vacc, voutput_min, vl);

        // 应用最大值约束
        vacc = vfmin_vf_f32m2(vacc, voutput_max, vl);

        // 存储结果
        vse32_v_f32m2(output, vacc, vl);
        output += vl;
        size -= vl;
    } while (size > 0);
}

//向量减法
void xnn_f32_vsub_minmax_ukernel__rvv_u2v(
        size_t batch,
        const float* input_a,
        const float* input_b,
        float* output,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(batch != 0);
    assert(batch % sizeof(float) == 0);
    assert(input_a != NULL);
    assert(input_b != NULL);
    assert(output != NULL);

    const float voutput_min = params->scalar.min;
    const float voutput_max = params->scalar.max;

    size_t size = batch / sizeof(float);
    do {
        // 动态设置向量长度
        const size_t vl = vsetvl_e32m2(size);

        // 加载输入向量
        vfloat32m2_t va = vle32_v_f32m2(input_a, vl);
        vfloat32m2_t vb = vle32_v_f32m2(input_b, vl);
        input_a += vl;
        input_b += vl;

        // 执行向量减法
        vfloat32m2_t vacc = vfsub_vv_f32m2(va, vb, vl);

        // 应用最小值约束
        vacc = vfmax_vf_f32m2(vacc, voutput_min, vl);

        // 应用最大值约束
        vacc = vfmin_vf_f32m2(vacc, voutput_max, vl);

        // 存储结果
        vse32_v_f32m2(output, vacc, vl);
        output += vl;
        size -= vl;
    } while (size > 0);
}

//向量乘法
void xnn_f32_vmul_minmax_ukernel__rvv_u2v(
        size_t batch,
        const float* input_a,
        const float* input_b,
        float* output,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(batch != 0);
    assert(batch % sizeof(float) == 0);
    assert(input_a != NULL);
    assert(input_b != NULL);
    assert(output != NULL);

    const float voutput_min = params->scalar.min;
    const float voutput_max = params->scalar.max;

    size_t size = batch / sizeof(float);
    do {
        // 动态设置向量长度
        const size_t vl = vsetvl_e32m2(size);

        // 加载输入向量
        vfloat32m2_t va = vle32_v_f32m2(input_a, vl);
        vfloat32m2_t vb = vle32_v_f32m2(input_b, vl);
        input_a += vl;
        input_b += vl;

        // 执行向量乘法
        vfloat32m2_t vacc = vfmul_vv_f32m2(va, vb, vl);

        // 应用最小值约束
        vacc = vfmax_vf_f32m2(vacc, voutput_min, vl);

        // 应用最大值约束
        vacc = vfmin_vf_f32m2(vacc, voutput_max, vl);

        // 存储结果
        vse32_v_f32m2(output, vacc, vl);
        output += vl;
        size -= vl;
    } while (size > 0);
}

//向量除法
void xnn_f32_vdivc_minmax_ukernel__rvv_u2v(
        size_t batch,
        const float* input_a,
        const float* input_b,
        float* output,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(batch != 0);
    assert(batch % sizeof(float) == 0);
    assert(input_a != NULL);
    assert(input_b != NULL);
    assert(output != NULL);

    const float voutput_min = params->scalar.min;
    const float voutput_max = params->scalar.max;
    const float vb = *input_b;

    size_t size = batch / sizeof(float);
    do {
        // 动态设置向量长度
        const size_t vl = vsetvl_e32m2(size);

        // 加载输入向量
        vfloat32m2_t va = vle32_v_f32m2(input_a, vl);
        input_a += vl;

        // 执行向量除法
        vfloat32m2_t vacc = vfdiv_vf_f32m2(va, vb, vl);

        // 应用最小值约束
        vacc = vfmax_vf_f32m2(vacc, voutput_min, vl);

        // 应用最大值约束
        vacc = vfmin_vf_f32m2(vacc, voutput_max, vl);

        // 存储结果
        vse32_v_f32m2(output, vacc, vl);
        output += vl;
        size -= vl;
    } while (size > 0);
}

//向量加法
void xnn_f32_vaddc_minmax_ukernel__rvv_u2v(
        size_t batch,
        const float* input_a,
        const float* input_b,
        float* output,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(batch != 0);
    assert(batch % sizeof(float) == 0);
    assert(input_a != NULL);
    assert(input_b != NULL);
    assert(output != NULL);

    const float voutput_min = params->scalar.min;
    const float voutput_max = params->scalar.max;
    const float vb = *input_b;

    size_t size = batch / sizeof(float);
    do {
        // 动态设置向量长度
        const size_t vl = vsetvl_e32m2(size);

        // 加载输入向量
        vfloat32m2_t va = vle32_v_f32m2(input_a, vl);
        input_a += vl;

        // 执行向量加法
        vfloat32m2_t vacc = vfadd_vf_f32m2(va, vb, vl);

        // 应用最小值约束
        vacc = vfmax_vf_f32m2(vacc, voutput_min, vl);

        // 应用最大值约束
        vacc = vfmin_vf_f32m2(vacc, voutput_max, vl);

        // 存储结果
        vse32_v_f32m2(output, vacc, vl);
        output += vl;
        size -= vl;
    } while (size > 0);
}

//向量减法
void xnn_f32_vsubc_minmax_ukernel__rvv_u2v(
        size_t batch,
        const float* input_a,
        const float* input_b,
        float* output,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(batch != 0);
    assert(batch % sizeof(float) == 0);
    assert(input_a != NULL);
    assert(input_b != NULL);
    assert(output != NULL);

    const float voutput_min = params->scalar.min;
    const float voutput_max = params->scalar.max;
    const float vb = *input_b;

    size_t size = batch / sizeof(float);
    do {
        // 动态设置向量长度
        const size_t vl = vsetvl_e32m2(size);

        // 加载输入向量
        vfloat32m2_t va = vle32_v_f32m2(input_a, vl);
        input_a += vl;

        // 执行向量减法
        vfloat32m2_t vacc = vfsub_vf_f32m2(va, vb, vl);

        // 应用最小值约束
        vacc = vfmax_vf_f32m2(vacc, voutput_min, vl);

        // 应用最大值约束
        vacc = vfmin_vf_f32m2(vacc, voutput_max, vl);

        // 存储结果
        vse32_v_f32m2(output, vacc, vl);
        output += vl;
        size -= vl;
    } while (size > 0);
}

//向量乘法
void xnn_f32_vmulc_minmax_ukernel__rvv_u2v(
        size_t batch,
        const float* input_a,
        const float* input_b,
        float* output,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(batch != 0);
    assert(batch % sizeof(float) == 0);
    assert(input_a != NULL);
    assert(input_b != NULL);
    assert(output != NULL);

    const float voutput_min = params->scalar.min;
    const float voutput_max = params->scalar.max;
    const float vb = *input_b;

    size_t size = batch / sizeof(float);
    do {
        // 动态设置向量长度
        const size_t vl = vsetvl_e32m2(size);

        // 加载输入向量
        vfloat32m2_t va = vle32_v_f32m2(input_a, vl);
        input_a += vl;

        // 执行向量乘法
        vfloat32m2_t vacc = vfmul_vf_f32m2(va, vb, vl);

        // 应用最小值约束
        vacc = vfmax_vf_f32m2(vacc, voutput_min, vl);

        // 应用最大值约束
        vacc = vfmin_vf_f32m2(vacc, voutput_max, vl);

        // 存储结果
        vse32_v_f32m2(output, vacc, vl);
        output += vl;
        size -= vl;
    } while (size > 0);
}

//向量除法
void xnn_f32_vrdivc_minmax_ukernel__rvv_u2v(
        size_t batch,
        const float* input_a,
        const float* input_b,
        float* output,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(batch != 0);
    assert(batch % sizeof(float) == 0);
    assert(input_a != NULL);
    assert(input_b != NULL);
    assert(output != NULL);

    const float voutput_min = params->scalar.min;
    const float voutput_max = params->scalar.max;
    const float vb = *input_b;

    size_t size = batch / sizeof(float);
    do {
        // 动态设置向量长度
        const size_t vl = vsetvl_e32m2(size);

        // 加载输入向量
        vfloat32m2_t va = vle32_v_f32m2(input_a, vl);
        input_a += vl;

        // 执行向量除法
        vfloat32m2_t vacc = vfrdiv_vf_f32m2(va, vb, vl);

        // 应用最小值约束
        vacc = vfmax_vf_f32m2(vacc, voutput_min, vl);

        // 应用最大值约束
        vacc = vfmin_vf_f32m2(vacc, voutput_max, vl);

        // 存储结果
        vse32_v_f32m2(output, vacc, vl);
        output += vl;
        size -= vl;
    } while (size > 0);
}

//向量减法
void xnn_f32_vrsubc_minmax_ukernel__rvv_u2v(
        size_t batch,
        const float* input_a,
        const float* input_b,
        float* output,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(batch != 0);
    assert(batch % sizeof(float) == 0);
    assert(input_a != NULL);
    assert(input_b != NULL);
    assert(output != NULL);

    const float voutput_min = params->scalar.min;
    const float voutput_max = params->scalar.max;
    const float vb = *input_b;

    size_t size = batch / sizeof(float);
    do {
        // 动态设置向量长度
        const size_t vl = vsetvl_e32m2(size);

        // 加载输入向量
        vfloat32m2_t va = vle32_v_f32m2(input_a, vl);
        input_a += vl;

        // 执行向量减法
        vfloat32m2_t vacc = vfrsub_vf_f32m2(va, vb, vl);

        // 应用最小值约束
        vacc = vfmax_vf_f32m2(vacc, voutput_min, vl);

        // 应用最大值约束
        vacc = vfmin_vf_f32m2(vacc, voutput_max, vl);

        // 存储结果
        vse32_v_f32m2(output, vacc, vl);
        output += vl;
        size -= vl;
    } while (size > 0);
}







// u1v

//向量除法
void xnn_f32_vdiv_minmax_ukernel__rvv_u1v(
        size_t batch,
        const float* input_a,
        const float* input_b,
        float* output,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(batch != 0);
    assert(batch % sizeof(float) == 0);
    assert(input_a != NULL);
    assert(input_b != NULL);
    assert(output != NULL);

    const float voutput_min = params->scalar.min;
    const float voutput_max = params->scalar.max;

    size_t size = batch / sizeof(float);
    do {
        // 动态设置向量长度
        const size_t vl = vsetvl_e32m1(size);

        // 加载输入向量
        vfloat32m1_t va = vle32_v_f32m1(input_a, vl);
        vfloat32m1_t vb = vle32_v_f32m1(input_b, vl);
        input_a += vl;
        input_b += vl;

        // 执行向量除法
        vfloat32m1_t vacc = vfdiv_vv_f32m1(va, vb, vl);

        // 应用最小值约束
        vacc = vfmax_vf_f32m1(vacc, voutput_min, vl);

        // 应用最大值约束
        vacc = vfmin_vf_f32m1(vacc, voutput_max, vl);

        // 存储结果
        vse32_v_f32m1(output, vacc, vl);
        output += vl;
        size -= vl;
    } while (size > 0);
}

//向量加法
void xnn_f32_vadd_minmax_ukernel__rvv_u1v(
        size_t batch,
        const float* input_a,
        const float* input_b,
        float* output,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(batch != 0);
    assert(batch % sizeof(float) == 0);
    assert(input_a != NULL);
    assert(input_b != NULL);
    assert(output != NULL);

    const float voutput_min = params->scalar.min;
    const float voutput_max = params->scalar.max;

    size_t size = batch / sizeof(float);
    do {
        // 动态设置向量长度
        const size_t vl = vsetvl_e32m1(size);

        // 加载输入向量
        vfloat32m1_t va = vle32_v_f32m1(input_a, vl);
        vfloat32m1_t vb = vle32_v_f32m1(input_b, vl);
        input_a += vl;
        input_b += vl;

        // 执行向量加法
        vfloat32m1_t vacc = vfadd_vv_f32m1(va, vb, vl);

        // 应用最小值约束
        vacc = vfmax_vf_f32m1(vacc, voutput_min, vl);

        // 应用最大值约束
        vacc = vfmin_vf_f32m1(vacc, voutput_max, vl);

        // 存储结果
        vse32_v_f32m1(output, vacc, vl);
        output += vl;
        size -= vl;
    } while (size > 0);
}

//向量减法
void xnn_f32_vsub_minmax_ukernel__rvv_u1v(
        size_t batch,
        const float* input_a,
        const float* input_b,
        float* output,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(batch != 0);
    assert(batch % sizeof(float) == 0);
    assert(input_a != NULL);
    assert(input_b != NULL);
    assert(output != NULL);

    const float voutput_min = params->scalar.min;
    const float voutput_max = params->scalar.max;

    size_t size = batch / sizeof(float);
    do {
        // 动态设置向量长度
        const size_t vl = vsetvl_e32m1(size);

        // 加载输入向量
        vfloat32m1_t va = vle32_v_f32m1(input_a, vl);
        vfloat32m1_t vb = vle32_v_f32m1(input_b, vl);
        input_a += vl;
        input_b += vl;

        // 执行向量减法
        vfloat32m1_t vacc = vfsub_vv_f32m1(va, vb, vl);

        // 应用最小值约束
        vacc = vfmax_vf_f32m1(vacc, voutput_min, vl);

        // 应用最大值约束
        vacc = vfmin_vf_f32m1(vacc, voutput_max, vl);

        // 存储结果
        vse32_v_f32m1(output, vacc, vl);
        output += vl;
        size -= vl;
    } while (size > 0);
}

//向量乘法
void xnn_f32_vmul_minmax_ukernel__rvv_u1v(
        size_t batch,
        const float* input_a,
        const float* input_b,
        float* output,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(batch != 0);
    assert(batch % sizeof(float) == 0);
    assert(input_a != NULL);
    assert(input_b != NULL);
    assert(output != NULL);

    const float voutput_min = params->scalar.min;
    const float voutput_max = params->scalar.max;

    size_t size = batch / sizeof(float);
    do {
        // 动态设置向量长度
        const size_t vl = vsetvl_e32m1(size);

        // 加载输入向量
        vfloat32m1_t va = vle32_v_f32m1(input_a, vl);
        vfloat32m1_t vb = vle32_v_f32m1(input_b, vl);
        input_a += vl;
        input_b += vl;

        // 执行向量乘法
        vfloat32m1_t vacc = vfmul_vv_f32m1(va, vb, vl);

        // 应用最小值约束
        vacc = vfmax_vf_f32m1(vacc, voutput_min, vl);

        // 应用最大值约束
        vacc = vfmin_vf_f32m1(vacc, voutput_max, vl);

        // 存储结果
        vse32_v_f32m1(output, vacc, vl);
        output += vl;
        size -= vl;
    } while (size > 0);
}

//向量除法
void xnn_f32_vdivc_minmax_ukernel__rvv_u1v(
        size_t batch,
        const float* input_a,
        const float* input_b,
        float* output,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(batch != 0);
    assert(batch % sizeof(float) == 0);
    assert(input_a != NULL);
    assert(input_b != NULL);
    assert(output != NULL);

    const float voutput_min = params->scalar.min;
    const float voutput_max = params->scalar.max;
    const float vb = *input_b;

    size_t size = batch / sizeof(float);
    do {
        // 动态设置向量长度
        const size_t vl = vsetvl_e32m1(size);

        // 加载输入向量
        vfloat32m1_t va = vle32_v_f32m1(input_a, vl);
        input_a += vl;

        // 执行向量除法
        vfloat32m1_t vacc = vfdiv_vf_f32m1(va, vb, vl);

        // 应用最小值约束
        vacc = vfmax_vf_f32m1(vacc, voutput_min, vl);

        // 应用最大值约束
        vacc = vfmin_vf_f32m1(vacc, voutput_max, vl);

        // 存储结果
        vse32_v_f32m1(output, vacc, vl);
        output += vl;
        size -= vl;
    } while (size > 0);
}

//向量加法
void xnn_f32_vaddc_minmax_ukernel__rvv_u1v(
        size_t batch,
        const float* input_a,
        const float* input_b,
        float* output,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(batch != 0);
    assert(batch % sizeof(float) == 0);
    assert(input_a != NULL);
    assert(input_b != NULL);
    assert(output != NULL);

    const float voutput_min = params->scalar.min;
    const float voutput_max = params->scalar.max;
    const float vb = *input_b;

    size_t size = batch / sizeof(float);
    do {
        // 动态设置向量长度
        const size_t vl = vsetvl_e32m1(size);

        // 加载输入向量
        vfloat32m1_t va = vle32_v_f32m1(input_a, vl);
        input_a += vl;

        // 执行向量加法
        vfloat32m1_t vacc = vfadd_vf_f32m1(va, vb, vl);

        // 应用最小值约束
        vacc = vfmax_vf_f32m1(vacc, voutput_min, vl);

        // 应用最大值约束
        vacc = vfmin_vf_f32m1(vacc, voutput_max, vl);

        // 存储结果
        vse32_v_f32m1(output, vacc, vl);
        output += vl;
        size -= vl;
    } while (size > 0);
}

//向量减法
void xnn_f32_vsubc_minmax_ukernel__rvv_u1v(
        size_t batch,
        const float* input_a,
        const float* input_b,
        float* output,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(batch != 0);
    assert(batch % sizeof(float) == 0);
    assert(input_a != NULL);
    assert(input_b != NULL);
    assert(output != NULL);

    const float voutput_min = params->scalar.min;
    const float voutput_max = params->scalar.max;
    const float vb = *input_b;

    size_t size = batch / sizeof(float);
    do {
        // 动态设置向量长度
        const size_t vl = vsetvl_e32m1(size);

        // 加载输入向量
        vfloat32m1_t va = vle32_v_f32m1(input_a, vl);
        input_a += vl;

        // 执行向量减法
        vfloat32m1_t vacc = vfsub_vf_f32m1(va, vb, vl);

        // 应用最小值约束
        vacc = vfmax_vf_f32m1(vacc, voutput_min, vl);

        // 应用最大值约束
        vacc = vfmin_vf_f32m1(vacc, voutput_max, vl);

        // 存储结果
        vse32_v_f32m1(output, vacc, vl);
        output += vl;
        size -= vl;
    } while (size > 0);
}

//向量乘法
void xnn_f32_vmulc_minmax_ukernel__rvv_u1v(
        size_t batch,
        const float* input_a,
        const float* input_b,
        float* output,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(batch != 0);
    assert(batch % sizeof(float) == 0);
    assert(input_a != NULL);
    assert(input_b != NULL);
    assert(output != NULL);

    const float voutput_min = params->scalar.min;
    const float voutput_max = params->scalar.max;
    const float vb = *input_b;

    size_t size = batch / sizeof(float);
    do {
        // 动态设置向量长度
        const size_t vl = vsetvl_e32m1(size);

        // 加载输入向量
        vfloat32m1_t va = vle32_v_f32m1(input_a, vl);
        input_a += vl;

        // 执行向量乘法
        vfloat32m1_t vacc = vfmul_vf_f32m1(va, vb, vl);

        // 应用最小值约束
        vacc = vfmax_vf_f32m1(vacc, voutput_min, vl);

        // 应用最大值约束
        vacc = vfmin_vf_f32m1(vacc, voutput_max, vl);

        // 存储结果
        vse32_v_f32m1(output, vacc, vl);
        output += vl;
        size -= vl;
    } while (size > 0);
}

//向量除法
void xnn_f32_vrdivc_minmax_ukernel__rvv_u1v(
        size_t batch,
        const float* input_a,
        const float* input_b,
        float* output,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(batch != 0);
    assert(batch % sizeof(float) == 0);
    assert(input_a != NULL);
    assert(input_b != NULL);
    assert(output != NULL);

    const float voutput_min = params->scalar.min;
    const float voutput_max = params->scalar.max;
    const float vb = *input_b;

    size_t size = batch / sizeof(float);
    do {
        // 动态设置向量长度
        const size_t vl = vsetvl_e32m1(size);

        // 加载输入向量
        vfloat32m1_t va = vle32_v_f32m1(input_a, vl);
        input_a += vl;

        // 执行向量除法
        vfloat32m1_t vacc = vfrdiv_vf_f32m1(va, vb, vl);

        // 应用最小值约束
        vacc = vfmax_vf_f32m1(vacc, voutput_min, vl);

        // 应用最大值约束
        vacc = vfmin_vf_f32m1(vacc, voutput_max, vl);

        // 存储结果
        vse32_v_f32m1(output, vacc, vl);
        output += vl;
        size -= vl;
    } while (size > 0);
}

//向量减法
void xnn_f32_vrsubc_minmax_ukernel__rvv_u1v(
        size_t batch,
        const float* input_a,
        const float* input_b,
        float* output,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(batch != 0);
    assert(batch % sizeof(float) == 0);
    assert(input_a != NULL);
    assert(input_b != NULL);
    assert(output != NULL);

    const float voutput_min = params->scalar.min;
    const float voutput_max = params->scalar.max;
    const float vb = *input_b;

    size_t size = batch / sizeof(float);
    do {
        // 动态设置向量长度
        const size_t vl = vsetvl_e32m1(size);

        // 加载输入向量
        vfloat32m1_t va = vle32_v_f32m1(input_a, vl);
        input_a += vl;

        // 执行向量减法
        vfloat32m1_t vacc = vfrsub_vf_f32m1(va, vb, vl);

        // 应用最小值约束
        vacc = vfmax_vf_f32m1(vacc, voutput_min, vl);

        // 应用最大值约束
        vacc = vfmin_vf_f32m1(vacc, voutput_max, vl);

        // 存储结果
        vse32_v_f32m1(output, vacc, vl);
        output += vl;
        size -= vl;
    } while (size > 0);
}



// u4v

//向量除法
void xnn_f32_vdiv_minmax_ukernel__rvv_u4v(
        size_t batch,
        const float* input_a,
        const float* input_b,
        float* output,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(batch != 0);
    assert(batch % sizeof(float) == 0);
    assert(input_a != NULL);
    assert(input_b != NULL);
    assert(output != NULL);

    const float voutput_min = params->scalar.min;
    const float voutput_max = params->scalar.max;

    size_t size = batch / sizeof(float);
    do {
        // 动态设置向量长度
        const size_t vl = vsetvl_e32m4(size);

        // 加载输入向量
        vfloat32m4_t va = vle32_v_f32m4(input_a, vl);
        vfloat32m4_t vb = vle32_v_f32m4(input_b, vl);
        input_a += vl;
        input_b += vl;

        // 执行向量除法
        vfloat32m4_t vacc = vfdiv_vv_f32m4(va, vb, vl);

        // 应用最小值约束
        vacc = vfmax_vf_f32m4(vacc, voutput_min, vl);

        // 应用最大值约束
        vacc = vfmin_vf_f32m4(vacc, voutput_max, vl);

        // 存储结果
        vse32_v_f32m4(output, vacc, vl);
        output += vl;
        size -= vl;
    } while (size > 0);
}

//向量加法
void xnn_f32_vadd_minmax_ukernel__rvv_u4v(
        size_t batch,
        const float* input_a,
        const float* input_b,
        float* output,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(batch != 0);
    assert(batch % sizeof(float) == 0);
    assert(input_a != NULL);
    assert(input_b != NULL);
    assert(output != NULL);

    const float voutput_min = params->scalar.min;
    const float voutput_max = params->scalar.max;

    size_t size = batch / sizeof(float);
    do {
        // 动态设置向量长度
        const size_t vl = vsetvl_e32m4(size);

        // 加载输入向量
        vfloat32m4_t va = vle32_v_f32m4(input_a, vl);
        vfloat32m4_t vb = vle32_v_f32m4(input_b, vl);
        input_a += vl;
        input_b += vl;

        // 执行向量加法
        vfloat32m4_t vacc = vfadd_vv_f32m4(va, vb, vl);

        // 应用最小值约束
        vacc = vfmax_vf_f32m4(vacc, voutput_min, vl);

        // 应用最大值约束
        vacc = vfmin_vf_f32m4(vacc, voutput_max, vl);

        // 存储结果
        vse32_v_f32m4(output, vacc, vl);
        output += vl;
        size -= vl;
    } while (size > 0);
}

//向量减法
void xnn_f32_vsub_minmax_ukernel__rvv_u4v(
        size_t batch,
        const float* input_a,
        const float* input_b,
        float* output,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(batch != 0);
    assert(batch % sizeof(float) == 0);
    assert(input_a != NULL);
    assert(input_b != NULL);
    assert(output != NULL);

    const float voutput_min = params->scalar.min;
    const float voutput_max = params->scalar.max;

    size_t size = batch / sizeof(float);
    do {
        // 动态设置向量长度
        const size_t vl = vsetvl_e32m4(size);

        // 加载输入向量
        vfloat32m4_t va = vle32_v_f32m4(input_a, vl);
        vfloat32m4_t vb = vle32_v_f32m4(input_b, vl);
        input_a += vl;
        input_b += vl;

        // 执行向量减法
        vfloat32m4_t vacc = vfsub_vv_f32m4(va, vb, vl);

        // 应用最小值约束
        vacc = vfmax_vf_f32m4(vacc, voutput_min, vl);

        // 应用最大值约束
        vacc = vfmin_vf_f32m4(vacc, voutput_max, vl);

        // 存储结果
        vse32_v_f32m4(output, vacc, vl);
        output += vl;
        size -= vl;
    } while (size > 0);
}

//向量乘法
void xnn_f32_vmul_minmax_ukernel__rvv_u4v(
        size_t batch,
        const float* input_a,
        const float* input_b,
        float* output,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(batch != 0);
    assert(batch % sizeof(float) == 0);
    assert(input_a != NULL);
    assert(input_b != NULL);
    assert(output != NULL);

    const float voutput_min = params->scalar.min;
    const float voutput_max = params->scalar.max;

    size_t size = batch / sizeof(float);
    do {
        // 动态设置向量长度
        const size_t vl = vsetvl_e32m4(size);

        // 加载输入向量
        vfloat32m4_t va = vle32_v_f32m4(input_a, vl);
        vfloat32m4_t vb = vle32_v_f32m4(input_b, vl);
        input_a += vl;
        input_b += vl;

        // 执行向量乘法
        vfloat32m4_t vacc = vfmul_vv_f32m4(va, vb, vl);

        // 应用最小值约束
        vacc = vfmax_vf_f32m4(vacc, voutput_min, vl);

        // 应用最大值约束
        vacc = vfmin_vf_f32m4(vacc, voutput_max, vl);

        // 存储结果
        vse32_v_f32m4(output, vacc, vl);
        output += vl;
        size -= vl;
    } while (size > 0);
}

//向量除法
void xnn_f32_vdivc_minmax_ukernel__rvv_u4v(
        size_t batch,
        const float* input_a,
        const float* input_b,
        float* output,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(batch != 0);
    assert(batch % sizeof(float) == 0);
    assert(input_a != NULL);
    assert(input_b != NULL);
    assert(output != NULL);

    const float voutput_min = params->scalar.min;
    const float voutput_max = params->scalar.max;
    const float vb = *input_b;

    size_t size = batch / sizeof(float);
    do {
        // 动态设置向量长度
        const size_t vl = vsetvl_e32m4(size);

        // 加载输入向量
        vfloat32m4_t va = vle32_v_f32m4(input_a, vl);
        input_a += vl;

        // 执行向量除法
        vfloat32m4_t vacc = vfdiv_vf_f32m4(va, vb, vl);

        // 应用最小值约束
        vacc = vfmax_vf_f32m4(vacc, voutput_min, vl);

        // 应用最大值约束
        vacc = vfmin_vf_f32m4(vacc, voutput_max, vl);

        // 存储结果
        vse32_v_f32m4(output, vacc, vl);
        output += vl;
        size -= vl;
    } while (size > 0);
}

//向量加法
void xnn_f32_vaddc_minmax_ukernel__rvv_u4v(
        size_t batch,
        const float* input_a,
        const float* input_b,
        float* output,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(batch != 0);
    assert(batch % sizeof(float) == 0);
    assert(input_a != NULL);
    assert(input_b != NULL);
    assert(output != NULL);

    const float voutput_min = params->scalar.min;
    const float voutput_max = params->scalar.max;
    const float vb = *input_b;

    size_t size = batch / sizeof(float);
    do {
        // 动态设置向量长度
        const size_t vl = vsetvl_e32m4(size);

        // 加载输入向量
        vfloat32m4_t va = vle32_v_f32m4(input_a, vl);
        input_a += vl;

        // 执行向量加法
        vfloat32m4_t vacc = vfadd_vf_f32m4(va, vb, vl);

        // 应用最小值约束
        vacc = vfmax_vf_f32m4(vacc, voutput_min, vl);

        // 应用最大值约束
        vacc = vfmin_vf_f32m4(vacc, voutput_max, vl);

        // 存储结果
        vse32_v_f32m4(output, vacc, vl);
        output += vl;
        size -= vl;
    } while (size > 0);
}

//向量减法
void xnn_f32_vsubc_minmax_ukernel__rvv_u4v(
        size_t batch,
        const float* input_a,
        const float* input_b,
        float* output,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(batch != 0);
    assert(batch % sizeof(float) == 0);
    assert(input_a != NULL);
    assert(input_b != NULL);
    assert(output != NULL);

    const float voutput_min = params->scalar.min;
    const float voutput_max = params->scalar.max;
    const float vb = *input_b;

    size_t size = batch / sizeof(float);
    do {
        // 动态设置向量长度
        const size_t vl = vsetvl_e32m4(size);

        // 加载输入向量
        vfloat32m4_t va = vle32_v_f32m4(input_a, vl);
        input_a += vl;

        // 执行向量减法
        vfloat32m4_t vacc = vfsub_vf_f32m4(va, vb, vl);

        // 应用最小值约束
        vacc = vfmax_vf_f32m4(vacc, voutput_min, vl);

        // 应用最大值约束
        vacc = vfmin_vf_f32m4(vacc, voutput_max, vl);

        // 存储结果
        vse32_v_f32m4(output, vacc, vl);
        output += vl;
        size -= vl;
    } while (size > 0);
}

//向量乘法
void xnn_f32_vmulc_minmax_ukernel__rvv_u4v(
        size_t batch,
        const float* input_a,
        const float* input_b,
        float* output,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(batch != 0);
    assert(batch % sizeof(float) == 0);
    assert(input_a != NULL);
    assert(input_b != NULL);
    assert(output != NULL);

    const float voutput_min = params->scalar.min;
    const float voutput_max = params->scalar.max;
    const float vb = *input_b;

    size_t size = batch / sizeof(float);
    do {
        // 动态设置向量长度
        const size_t vl = vsetvl_e32m4(size);

        // 加载输入向量
        vfloat32m4_t va = vle32_v_f32m4(input_a, vl);
        input_a += vl;

        // 执行向量乘法
        vfloat32m4_t vacc = vfmul_vf_f32m4(va, vb, vl);

        // 应用最小值约束
        vacc = vfmax_vf_f32m4(vacc, voutput_min, vl);

        // 应用最大值约束
        vacc = vfmin_vf_f32m4(vacc, voutput_max, vl);

        // 存储结果
        vse32_v_f32m4(output, vacc, vl);
        output += vl;
        size -= vl;
    } while (size > 0);
}

//向量除法
void xnn_f32_vrdivc_minmax_ukernel__rvv_u4v(
        size_t batch,
        const float* input_a,
        const float* input_b,
        float* output,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(batch != 0);
    assert(batch % sizeof(float) == 0);
    assert(input_a != NULL);
    assert(input_b != NULL);
    assert(output != NULL);

    const float voutput_min = params->scalar.min;
    const float voutput_max = params->scalar.max;
    const float vb = *input_b;

    size_t size = batch / sizeof(float);
    do {
        // 动态设置向量长度
        const size_t vl = vsetvl_e32m4(size);

        // 加载输入向量
        vfloat32m4_t va = vle32_v_f32m4(input_a, vl);
        input_a += vl;

        // 执行向量除法
        vfloat32m4_t vacc = vfrdiv_vf_f32m4(va, vb, vl);

        // 应用最小值约束
        vacc = vfmax_vf_f32m4(vacc, voutput_min, vl);

        // 应用最大值约束
        vacc = vfmin_vf_f32m4(vacc, voutput_max, vl);

        // 存储结果
        vse32_v_f32m4(output, vacc, vl);
        output += vl;
        size -= vl;
    } while (size > 0);
}

//向量减法
void xnn_f32_vrsubc_minmax_ukernel__rvv_u4v(
        size_t batch,
        const float* input_a,
        const float* input_b,
        float* output,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(batch != 0);
    assert(batch % sizeof(float) == 0);
    assert(input_a != NULL);
    assert(input_b != NULL);
    assert(output != NULL);

    const float voutput_min = params->scalar.min;
    const float voutput_max = params->scalar.max;
    const float vb = *input_b;

    size_t size = batch / sizeof(float);
    do {
        // 动态设置向量长度
        const size_t vl = vsetvl_e32m4(size);

        // 加载输入向量
        vfloat32m4_t va = vle32_v_f32m4(input_a, vl);
        input_a += vl;

        // 执行向量减法
        vfloat32m4_t vacc = vfrsub_vf_f32m4(va, vb, vl);

        // 应用最小值约束
        vacc = vfmax_vf_f32m4(vacc, voutput_min, vl);

        // 应用最大值约束
        vacc = vfmin_vf_f32m4(vacc, voutput_max, vl);

        // 存储结果
        vse32_v_f32m4(output, vacc, vl);
        output += vl;
        size -= vl;
    } while (size > 0);
}



// u8v

//向量除法
void xnn_f32_vdiv_minmax_ukernel__rvv_u8v(
        size_t batch,
        const float* input_a,
        const float* input_b,
        float* output,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(batch != 0);
    assert(batch % sizeof(float) == 0);
    assert(input_a != NULL);
    assert(input_b != NULL);
    assert(output != NULL);

    const float voutput_min = params->scalar.min;
    const float voutput_max = params->scalar.max;

    size_t size = batch / sizeof(float);
    do {
        // 动态设置向量长度
        const size_t vl = vsetvl_e32m8(size);

        // 加载输入向量
        vfloat32m8_t va = vle32_v_f32m8(input_a, vl);
        vfloat32m8_t vb = vle32_v_f32m8(input_b, vl);
        input_a += vl;
        input_b += vl;

        // 执行向量除法
        vfloat32m8_t vacc = vfdiv_vv_f32m8(va, vb, vl);

        // 应用最小值约束
        vacc = vfmax_vf_f32m8(vacc, voutput_min, vl);

        // 应用最大值约束
        vacc = vfmin_vf_f32m8(vacc, voutput_max, vl);

        // 存储结果
        vse32_v_f32m8(output, vacc, vl);
        output += vl;
        size -= vl;
    } while (size > 0);
}

//向量加法
void xnn_f32_vadd_minmax_ukernel__rvv_u8v(
        size_t batch,
        const float* input_a,
        const float* input_b,
        float* output,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(batch != 0);
    assert(batch % sizeof(float) == 0);
    assert(input_a != NULL);
    assert(input_b != NULL);
    assert(output != NULL);

    const float voutput_min = params->scalar.min;
    const float voutput_max = params->scalar.max;

    size_t size = batch / sizeof(float);
    do {
        // 动态设置向量长度
        const size_t vl = vsetvl_e32m8(size);

        // 加载输入向量
        vfloat32m8_t va = vle32_v_f32m8(input_a, vl);
        vfloat32m8_t vb = vle32_v_f32m8(input_b, vl);
        input_a += vl;
        input_b += vl;

        // 执行向量加法
        vfloat32m8_t vacc = vfadd_vv_f32m8(va, vb, vl);

        // 应用最小值约束
        vacc = vfmax_vf_f32m8(vacc, voutput_min, vl);

        // 应用最大值约束
        vacc = vfmin_vf_f32m8(vacc, voutput_max, vl);

        // 存储结果
        vse32_v_f32m8(output, vacc, vl);
        output += vl;
        size -= vl;
    } while (size > 0);
}

//向量减法
void xnn_f32_vsub_minmax_ukernel__rvv_u8v(
        size_t batch,
        const float* input_a,
        const float* input_b,
        float* output,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(batch != 0);
    assert(batch % sizeof(float) == 0);
    assert(input_a != NULL);
    assert(input_b != NULL);
    assert(output != NULL);

    const float voutput_min = params->scalar.min;
    const float voutput_max = params->scalar.max;

    size_t size = batch / sizeof(float);
    do {
        // 动态设置向量长度
        const size_t vl = vsetvl_e32m8(size);

        // 加载输入向量
        vfloat32m8_t va = vle32_v_f32m8(input_a, vl);
        vfloat32m8_t vb = vle32_v_f32m8(input_b, vl);
        input_a += vl;
        input_b += vl;

        // 执行向量减法
        vfloat32m8_t vacc = vfsub_vv_f32m8(va, vb, vl);

        // 应用最小值约束
        vacc = vfmax_vf_f32m8(vacc, voutput_min, vl);

        // 应用最大值约束
        vacc = vfmin_vf_f32m8(vacc, voutput_max, vl);

        // 存储结果
        vse32_v_f32m8(output, vacc, vl);
        output += vl;
        size -= vl;
    } while (size > 0);
}

//向量乘法
void xnn_f32_vmul_minmax_ukernel__rvv_u8v(
        size_t batch,
        const float* input_a,
        const float* input_b,
        float* output,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(batch != 0);
    assert(batch % sizeof(float) == 0);
    assert(input_a != NULL);
    assert(input_b != NULL);
    assert(output != NULL);

    const float voutput_min = params->scalar.min;
    const float voutput_max = params->scalar.max;

    size_t size = batch / sizeof(float);
    do {
        // 动态设置向量长度
        const size_t vl = vsetvl_e32m8(size);

        // 加载输入向量
        vfloat32m8_t va = vle32_v_f32m8(input_a, vl);
        vfloat32m8_t vb = vle32_v_f32m8(input_b, vl);
        input_a += vl;
        input_b += vl;

        // 执行向量乘法
        vfloat32m8_t vacc = vfmul_vv_f32m8(va, vb, vl);

        // 应用最小值约束
        vacc = vfmax_vf_f32m8(vacc, voutput_min, vl);

        // 应用最大值约束
        vacc = vfmin_vf_f32m8(vacc, voutput_max, vl);

        // 存储结果
        vse32_v_f32m8(output, vacc, vl);
        output += vl;
        size -= vl;
    } while (size > 0);
}

//向量除法
void xnn_f32_vdivc_minmax_ukernel__rvv_u8v(
        size_t batch,
        const float* input_a,
        const float* input_b,
        float* output,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(batch != 0);
    assert(batch % sizeof(float) == 0);
    assert(input_a != NULL);
    assert(input_b != NULL);
    assert(output != NULL);

    const float voutput_min = params->scalar.min;
    const float voutput_max = params->scalar.max;
    const float vb = *input_b;

    size_t size = batch / sizeof(float);
    do {
        // 动态设置向量长度
        const size_t vl = vsetvl_e32m8(size);

        // 加载输入向量
        vfloat32m8_t va = vle32_v_f32m8(input_a, vl);
        input_a += vl;

        // 执行向量除法
        vfloat32m8_t vacc = vfdiv_vf_f32m8(va, vb, vl);

        // 应用最小值约束
        vacc = vfmax_vf_f32m8(vacc, voutput_min, vl);

        // 应用最大值约束
        vacc = vfmin_vf_f32m8(vacc, voutput_max, vl);

        // 存储结果
        vse32_v_f32m8(output, vacc, vl);
        output += vl;
        size -= vl;
    } while (size > 0);
}

//向量加法
void xnn_f32_vaddc_minmax_ukernel__rvv_u8v(
        size_t batch,
        const float* input_a,
        const float* input_b,
        float* output,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(batch != 0);
    assert(batch % sizeof(float) == 0);
    assert(input_a != NULL);
    assert(input_b != NULL);
    assert(output != NULL);

    const float voutput_min = params->scalar.min;
    const float voutput_max = params->scalar.max;
    const float vb = *input_b;

    size_t size = batch / sizeof(float);
    do {
        // 动态设置向量长度
        const size_t vl = vsetvl_e32m8(size);

        // 加载输入向量
        vfloat32m8_t va = vle32_v_f32m8(input_a, vl);
        input_a += vl;

        // 执行向量加法
        vfloat32m8_t vacc = vfadd_vf_f32m8(va, vb, vl);

        // 应用最小值约束
        vacc = vfmax_vf_f32m8(vacc, voutput_min, vl);

        // 应用最大值约束
        vacc = vfmin_vf_f32m8(vacc, voutput_max, vl);

        // 存储结果
        vse32_v_f32m8(output, vacc, vl);
        output += vl;
        size -= vl;
    } while (size > 0);
}

//向量减法
void xnn_f32_vsubc_minmax_ukernel__rvv_u8v(
        size_t batch,
        const float* input_a,
        const float* input_b,
        float* output,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(batch != 0);
    assert(batch % sizeof(float) == 0);
    assert(input_a != NULL);
    assert(input_b != NULL);
    assert(output != NULL);

    const float voutput_min = params->scalar.min;
    const float voutput_max = params->scalar.max;
    const float vb = *input_b;

    size_t size = batch / sizeof(float);
    do {
        // 动态设置向量长度
        const size_t vl = vsetvl_e32m8(size);

        // 加载输入向量
        vfloat32m8_t va = vle32_v_f32m8(input_a, vl);
        input_a += vl;

        // 执行向量减法
        vfloat32m8_t vacc = vfsub_vf_f32m8(va, vb, vl);

        // 应用最小值约束
        vacc = vfmax_vf_f32m8(vacc, voutput_min, vl);

        // 应用最大值约束
        vacc = vfmin_vf_f32m8(vacc, voutput_max, vl);

        // 存储结果
        vse32_v_f32m8(output, vacc, vl);
        output += vl;
        size -= vl;
    } while (size > 0);
}

//向量乘法
void xnn_f32_vmulc_minmax_ukernel__rvv_u8v(
        size_t batch,
        const float* input_a,
        const float* input_b,
        float* output,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(batch != 0);
    assert(batch % sizeof(float) == 0);
    assert(input_a != NULL);
    assert(input_b != NULL);
    assert(output != NULL);

    const float voutput_min = params->scalar.min;
    const float voutput_max = params->scalar.max;
    const float vb = *input_b;

    size_t size = batch / sizeof(float);
    do {
        // 动态设置向量长度
        const size_t vl = vsetvl_e32m8(size);

        // 加载输入向量
        vfloat32m8_t va = vle32_v_f32m8(input_a, vl);
        input_a += vl;

        // 执行向量乘法
        vfloat32m8_t vacc = vfmul_vf_f32m8(va, vb, vl);

        // 应用最小值约束
        vacc = vfmax_vf_f32m8(vacc, voutput_min, vl);

        // 应用最大值约束
        vacc = vfmin_vf_f32m8(vacc, voutput_max, vl);

        // 存储结果
        vse32_v_f32m8(output, vacc, vl);
        output += vl;
        size -= vl;
    } while (size > 0);
}

//向量除法
void xnn_f32_vrdivc_minmax_ukernel__rvv_u8v(
        size_t batch,
        const float* input_a,
        const float* input_b,
        float* output,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(batch != 0);
    assert(batch % sizeof(float) == 0);
    assert(input_a != NULL);
    assert(input_b != NULL);
    assert(output != NULL);

    const float voutput_min = params->scalar.min;
    const float voutput_max = params->scalar.max;
    const float vb = *input_b;

    size_t size = batch / sizeof(float);
    do {
        // 动态设置向量长度
        const size_t vl = vsetvl_e32m8(size);

        // 加载输入向量
        vfloat32m8_t va = vle32_v_f32m8(input_a, vl);
        input_a += vl;

        // 执行向量除法
        vfloat32m8_t vacc = vfrdiv_vf_f32m8(va, vb, vl);

        // 应用最小值约束
        vacc = vfmax_vf_f32m8(vacc, voutput_min, vl);

        // 应用最大值约束
        vacc = vfmin_vf_f32m8(vacc, voutput_max, vl);

        // 存储结果
        vse32_v_f32m8(output, vacc, vl);
        output += vl;
        size -= vl;
    } while (size > 0);
}

//向量减法
void xnn_f32_vrsubc_minmax_ukernel__rvv_u8v(
        size_t batch,
        const float* input_a,
        const float* input_b,
        float* output,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(batch != 0);
    assert(batch % sizeof(float) == 0);
    assert(input_a != NULL);
    assert(input_b != NULL);
    assert(output != NULL);

    const float voutput_min = params->scalar.min;
    const float voutput_max = params->scalar.max;
    const float vb = *input_b;

    size_t size = batch / sizeof(float);
    do {
        // 动态设置向量长度
        const size_t vl = vsetvl_e32m8(size);

        // 加载输入向量
        vfloat32m8_t va = vle32_v_f32m8(input_a, vl);
        input_a += vl;

        // 执行向量减法
        vfloat32m8_t vacc = vfrsub_vf_f32m8(va, vb, vl);

        // 应用最小值约束
        vacc = vfmax_vf_f32m8(vacc, voutput_min, vl);

        // 应用最大值约束
        vacc = vfmin_vf_f32m8(vacc, voutput_max, vl);

        // 存储结果
        vse32_v_f32m8(output, vacc, vl);
        output += vl;
        size -= vl;
    } while (size > 0);
}





void xnn_f32_igemm_ukernel_1x4__rvv_u1v(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(mr != 0);
    assert(mr <= 1);
    assert(nc != 0);
    assert(kc != 0);
    assert(kc % sizeof(float) == 0);
    assert(ks != 0);
    assert(ks % (1 * sizeof(void*)) == 0);
    assert(a_offset % sizeof(float) == 0);
    assert(a != NULL);
    assert(w != NULL);
    assert(c != NULL);

    float* c0 = c;

    do {
        size_t vl = vsetvl_e32m1(nc); // vector length
        vfloat32m1_t vacc0 = vle32_v_f32m1(w, 4); // 1st row count
        w += 4;

         size_t p = ks;
        size_t kcl = kc / sizeof(float);
        do {
             const float* restrict a0 = a[0];
            assert(a0 != NULL);
            if XNN_UNPREDICTABLE(a0 != zero) {
                a0 = (const float*) ((uintptr_t) a0 + a_offset);
            }
            a += 1;

            size_t k = kc;
            for(size_t k = 0; k < kcl ; k++){
                vfloat32m1_t vw = vle32_v_f32m1(w, 4);
                w += 4;
                vacc0 = vfmacc_vf_f32m1(vacc0, *a0, vw, 4); // update 1st row count
                a0++;
            }
            p -= 1 * sizeof(void*);
        } while (p != 0);
        vse32_v_f32m1(c0, vacc0, vl); // store 1st row result

        if XNN_LIKELY(nc >= 4) {
            c0 = (float*) ((uintptr_t) c0 + cn_stride);
            a = (const float**restrict) ((uintptr_t) a - ks);
        }
        nc -= vl;
    } while (nc != 0);
}




void xnn_f32_igemm_ukernel_1x8__rvv_u2v(
        size_t mr,
        size_t nc,
        size_t kc,
        size_t ks,
        const float** restrict a,
        const float* restrict w,
        float* restrict c,
        size_t cm_stride,
        size_t cn_stride,
        size_t a_offset,
        const float* zero,
        const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(mr != 0);
    assert(mr <= 1);
    assert(nc != 0);
    assert(kc != 0);
    assert(kc % sizeof(float) == 0);
    assert(ks != 0);
    assert(ks % (1 * sizeof(void*)) == 0);
    assert(a_offset % sizeof(float) == 0);
    assert(a != NULL);
    assert(w != NULL);
    assert(c != NULL);

    float* c0 = c;

    do {
        size_t vl = vsetvl_e32m2(nc); // vector length
        vfloat32m2_t vacc0 = vle32_v_f32m2(w, 8); // 1st row count
        w += 8;

        size_t p = ks;
        size_t kcl = kc / sizeof(float);
        do {
            const float* restrict a0 = a[0];
            assert(a0 != NULL);
            if XNN_UNPREDICTABLE(a0 != zero) {
                a0 = (const float*) ((uintptr_t) a0 + a_offset);
            }
            a += 1;

            size_t k = kc;
            for(size_t k = 0; k < kcl ; k++){
                vfloat32m2_t vw = vle32_v_f32m2(w, 8);
                w += 8;
                vacc0 = vfmacc_vf_f32m2(vacc0, *a0, vw, 8); // update 1st row count
                a0++;
            }
            p -= 1 * sizeof(void*);
        } while (p != 0);
        vse32_v_f32m2(c0, vacc0, vl); // store 1st row result

        if XNN_LIKELY(nc >= 4) {
            c0 = (float*) ((uintptr_t) c0 + cn_stride);
            a = (const float**restrict) ((uintptr_t) a - ks);
        }
        nc -= vl;
    } while (nc != 0);
}


void xnn_f32_igemm_ukernel_4x8__rvv_u2v(
        size_t mr,
        size_t nc,
        size_t kc,
        size_t ks,
        const float** restrict a,
        const float* restrict w,
        float* restrict c,
        size_t cm_stride,
        size_t cn_stride,
        size_t a_offset,const float* zero,
        const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(mr != 0);
    assert(mr <= 4);
    assert(nc != 0);
    assert(kc != 0);
    assert(kc % sizeof(float) == 0);
    assert(ks != 0);
    assert(ks % (4 * sizeof(void*)) == 0);
    assert(a_offset % sizeof(float) == 0);
    assert(a != NULL);
    assert(w != NULL);
    assert(c != NULL);

    float* c0 = c;
    float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
    if XNN_UNPREDICTABLE(mr < 2) {
        c1 = c0;
    }
    float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
    if XNN_UNPREDICTABLE(mr <= 2) {
        c2 = c1;
    }
    float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
    if XNN_UNPREDICTABLE(mr != 4) {
        c3 = c2;
    }

    do {
        size_t vl = vsetvl_e32m2(nc); // vector length
        vfloat32m2_t vacc0 = vle32_v_f32m2(w, 8); // 1st row count
        vfloat32m2_t vacc1 = vle32_v_f32m2(w, 8); // 1st row count
        vfloat32m2_t vacc2 = vle32_v_f32m2(w, 8); // 1st row count
        vfloat32m2_t vacc3 = vle32_v_f32m2(w, 8); // 1st row count

        w += 8;

        size_t p = ks;
        size_t kcl = kc / sizeof(float);
        do {
            const float* restrict a0 = a[0];
            assert(a0 != NULL);
            if XNN_UNPREDICTABLE(a0 != zero) {
                a0 = (const float*) ((uintptr_t) a0 + a_offset);
            }
            const float* restrict a1 = a[1];
            assert(a1 != NULL);
            if XNN_UNPREDICTABLE(a1 != zero) {
                a1 = (const float*) ((uintptr_t) a1 + a_offset);
            }
            const float* restrict a2 = a[2];
            assert(a2 != NULL);
            if XNN_UNPREDICTABLE(a2 != zero) {
                a2 = (const float*) ((uintptr_t) a2 + a_offset);
            }
            const float* restrict a3 = a[3];
            assert(a3 != NULL);
            if XNN_UNPREDICTABLE(a3 != zero) {
                a3 = (const float*) ((uintptr_t) a3 + a_offset);
            }
            a += 4;

            size_t k = kc;
            for(size_t k = 0; k < kcl ; k++){
                vfloat32m2_t vw = vle32_v_f32m2(w, 8);
                w += 8;
                vacc0 = vfmacc_vf_f32m2(vacc0, *a0, vw, 8); // update 1st row count
                vacc1 = vfmacc_vf_f32m2(vacc1, *a1, vw, 8); // update 1st row count
                vacc2 = vfmacc_vf_f32m2(vacc2, *a2, vw, 8); // update 1st row count
                vacc3 = vfmacc_vf_f32m2(vacc3, *a3, vw, 8); // update 1st row count
                a0++;
                a1++;
                a2++;
                a3++;
            }
            p -= 4 * sizeof(void*);
        } while (p != 0);
        vse32_v_f32m2(c0, vacc0, vl); // store 1st row result
        vse32_v_f32m2(c1, vacc1, vl); // store 1st row result
        vse32_v_f32m2(c2, vacc2, vl); // store 1st row result
        vse32_v_f32m2(c3, vacc3, vl); // store 1st row result

        if XNN_LIKELY(nc >= 4) {
            c3 = (float*) ((uintptr_t) c3 + cn_stride);
            c2 = (float*) ((uintptr_t) c2 + cn_stride);
            c1 = (float*) ((uintptr_t) c1 + cn_stride);
            c0 = (float*) ((uintptr_t) c0 + cn_stride);

            a = (const float**restrict) ((uintptr_t) a - ks);
        }
        nc -= vl;
    } while (nc != 0);
}

void xnn_f32_igemm_ukernel_1x16__rvv_u2v(
        size_t mr,
        size_t nc,
        size_t kc,
        size_t ks,
        const float** restrict a,
        const float* restrict w,
        float* restrict c,
        size_t cm_stride,
        size_t cn_stride,
        size_t a_offset,
        const float* zero,
        const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(mr != 0);
    assert(mr <= 1);
    assert(nc != 0);
    assert(kc != 0);
    assert(kc % sizeof(float) == 0);
    assert(ks != 0);
    assert(ks % (1 * sizeof(void*)) == 0);
    assert(a_offset % sizeof(float) == 0);
    assert(a!= NULL);
    assert(w!= NULL);
    assert(c!= NULL);

    float* c0 = c;

    do {
        size_t vl = vsetvl_e32m4(nc); // vector length
        vfloat32m4_t vacc0 = vle32_v_f32m4(w, 16); // 1st row count
        w += 16;

        size_t p = ks;
        size_t kcl = kc / sizeof(float);
        do {
            const float* restrict a0 = a[0];
            assert(a0 != NULL);
            if XNN_UNPREDICTABLE(a0 != zero) {
                a0 = (const float*) ((uintptr_t) a0 + a_offset);
            }
            a += 1;

            size_t k = kc;
            for(size_t k = 0; k < kcl ; k++){
                vfloat32m4_t vw = vle32_v_f32m4(w, 16);
                w += 16;
                vacc0 = vfmacc_vf_f32m4(vacc0, *a0, vw, 16); // update 1st row count
                a0++;
            }
            p -= 1 * sizeof(void*);
        } while (p != 0);
        vse32_v_f32m4(c0, vacc0, vl); // store 1st row result

        if XNN_LIKELY(nc >= 4) {
            c0 = (float*) ((uintptr_t) c0 + cn_stride);
            a = (const float**restrict) ((uintptr_t) a - ks);
        }
        nc -= vl;
    } while (nc != 0);
}





void xnn_f32_igemm_ukernel_4x32__rvv_u8v(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,const float* zero,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(mr != 0);
    assert(mr <= 4);
    assert(nc != 0);
    assert(kc != 0);
    assert(kc % sizeof(float) == 0);
    assert(ks != 0);
    assert(ks % (4 * sizeof(void*)) == 0);
    assert(a_offset % sizeof(float) == 0);
    assert(a != NULL);
    assert(w != NULL);
    assert(c != NULL);

    float* c0 = c;
    float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
    if XNN_UNPREDICTABLE(mr < 2) {
        c1 = c0;
    }
    float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
    if XNN_UNPREDICTABLE(mr <= 2) {
        c2 = c1;
    }
    float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
    if XNN_UNPREDICTABLE(mr != 4) {
        c3 = c2;
    }

    do {
        size_t vl = vsetvl_e32m8(nc); // vector length
        vfloat32m8_t vacc0 = vle32_v_f32m8(w, 32); // 1st row count
        vfloat32m8_t vacc1 = vle32_v_f32m8(w, 32); // 1st row count
        vfloat32m8_t vacc2 = vle32_v_f32m8(w, 32); // 1st row count
        vfloat32m8_t vacc3 = vle32_v_f32m8(w, 32); // 1st row count

        w += 32;

        size_t p = ks;
        size_t kcl = kc / sizeof(float);
        do {
            const float* restrict a0 = a[0];
            assert(a0 != NULL);
            if XNN_UNPREDICTABLE(a0 != zero) {
                a0 = (const float*) ((uintptr_t) a0 + a_offset);
            }
            const float* restrict a1 = a[1];
            assert(a1 != NULL);
            if XNN_UNPREDICTABLE(a1 != zero) {
                a1 = (const float*) ((uintptr_t) a1 + a_offset);
            }
            const float* restrict a2 = a[2];
            assert(a2 != NULL);
            if XNN_UNPREDICTABLE(a2 != zero) {
                a2 = (const float*) ((uintptr_t) a2 + a_offset);
            }
            const float* restrict a3 = a[3];
            assert(a3 != NULL);
            if XNN_UNPREDICTABLE(a3 != zero) {
                a3 = (const float*) ((uintptr_t) a3 + a_offset);
            }
            a += 4;

            size_t k = kc;
            for(size_t k = 0; k < kcl ; k++){
                vfloat32m8_t vw = vle32_v_f32m8(w, 32);
                w += 32;
                vacc0 = vfmacc_vf_f32m8(vacc0, *a0, vw, 32); // update 1st row count
                vacc1 = vfmacc_vf_f32m8(vacc1, *a1, vw, 32); // update 1st row count
                vacc2 = vfmacc_vf_f32m8(vacc2, *a2, vw, 32); // update 1st row count
                vacc3 = vfmacc_vf_f32m8(vacc3, *a3, vw, 32); // update 1st row count
                a0++;
                a1++;
                a2++;
                a3++;
            }
            p -= 4 * sizeof(void*);
        } while (p != 0);
        vse32_v_f32m8(c0, vacc0, vl); // store 1st row result
        vse32_v_f32m8(c1, vacc1, vl); // store 1st row result
        vse32_v_f32m8(c2, vacc2, vl); // store 1st row result
        vse32_v_f32m8(c3, vacc3, vl); // store 1st row result

        if XNN_LIKELY(nc >= 4) {
            c3 = (float*) ((uintptr_t) c3 + cn_stride);
            c2 = (float*) ((uintptr_t) c2 + cn_stride);
            c1 = (float*) ((uintptr_t) c1 + cn_stride);
            c0 = (float*) ((uintptr_t) c0 + cn_stride);

            a = (const float**restrict) ((uintptr_t) a - ks);
        }
        nc -= vl;
    } while (nc != 0);
}



void xnn_f32_gemm_relu_ukernel_1x4__rvv_u1v(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
	assert(mr != 0);
	assert(mr <= 1);
	assert(nc != 0);
	assert(kc != 0);
	assert(kc % sizeof(float) == 0);
	assert(a != NULL);
	assert(w != NULL);
	assert(c != NULL);

	const float* a0 = a;
	float* c0 = c;
	size_t kcl = kc / sizeof(float);

	do {
		size_t vl = vsetvl_e32m1(nc);
		vfloat32m1_t vacc = vle32_v_f32m1(w, 4);
		w += 4;
		for(size_t k = 0; k < kcl ; k++){
			vfloat32m1_t vw = vle32_v_f32m1(w, 4);
			w += 4;
			vacc = vfmacc_vf_f32m1(vacc, *a0, vw, 4);
			a0++;
		}
        vacc = vfmax_vf_f32m1(vacc, 0.0f, vl);
		vse32_v_f32m1(c0, vacc, vl);
		if(nc >= 4){
      		c0 = (float*) ((uintptr_t) c0 + cn_stride);
      		a0 = (const void*) ((uintptr_t) a0 - kc);
		}
		nc -= vl;
	} while (nc != 0);
}

void xnn_f32_gemm_relu_ukernel_4x4__rvv_u1v(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(mr != 0);
    assert(mr <= 4); // max process 1 row
    assert(nc != 0);
    assert(kc != 0);
    assert(kc % sizeof(float) == 0);
    assert(a != NULL);
    assert(w != NULL);
    assert(c != NULL);

    const float* a0 = a;
    float* c0 = c;
    const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
    float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
    if XNN_UNPREDICTABLE(mr < 2) {
        a1 = a0;
        c1 = c0;
    }
    const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
    float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
    if XNN_UNPREDICTABLE(mr <= 2) {
        a2 = a1;
        c2 = c1;
    }
    const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
    float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
    if XNN_UNPREDICTABLE(mr != 4) {
        a3 = a2;
        c3 = c2;
    }

    size_t kcl = kc / sizeof(float);

    do {
        size_t vl = vsetvl_e32m1(nc); // vector length
        vfloat32m1_t vacc0 = vle32_v_f32m1(w, 4); // 1st row count
        vfloat32m1_t vacc1 = vle32_v_f32m1(w, 4); // 1st row count
        vfloat32m1_t vacc2 = vle32_v_f32m1(w, 4); // 1st row count
        vfloat32m1_t vacc3 = vle32_v_f32m1(w, 4); // 1st row count
        w += 4;
        for(size_t k = 0; k < kcl ; k++){
            vfloat32m1_t vw = vle32_v_f32m1(w, 4);
            w += 4;
            vacc0 = vfmacc_vf_f32m1(vacc0, *a0, vw, 4); // update 1st row count
            vacc1 = vfmacc_vf_f32m1(vacc1, *a1, vw, 4); // update 1st row count
            vacc2 = vfmacc_vf_f32m1(vacc2, *a2, vw, 4); // update 1st row count
            vacc3 = vfmacc_vf_f32m1(vacc3, *a3, vw, 4); // update 1st row count
            a0++;
            a1++;
            a2++;
            a3++;
        }
        vacc0 = vfmax_vf_f32m1(vacc0, 0.0f, vl);
        vacc1 = vfmax_vf_f32m1(vacc1, 0.0f, vl);
        vacc2 = vfmax_vf_f32m1(vacc2, 0.0f, vl);
        vacc3 = vfmax_vf_f32m1(vacc3, 0.0f, vl);
        vse32_v_f32m1(c0, vacc0, vl); // store 1st row result
        vse32_v_f32m1(c1, vacc1, vl); // store 1st row result
        vse32_v_f32m1(c2, vacc2, vl); // store 1st row result
        vse32_v_f32m1(c3, vacc3, vl); // store 1st row result
        if(nc >= 4){
            c0 = (float*) ((uintptr_t) c0 + cn_stride); // update 1st row matrix C pointer
            c1 = (float*) ((uintptr_t) c1 + cn_stride); // update 1st row matrix C pointer
            c2 = (float*) ((uintptr_t) c2 + cn_stride); // update 1st row matrix C pointer
            c3 = (float*) ((uintptr_t) c3 + cn_stride); // update 1st row matrix C pointer
            a0 = (const void*) ((uintptr_t) a0 - kc); // update 1st row matrix A pointer
            a1 = (const void*) ((uintptr_t) a1 - kc); // update 1st row matrix A pointer
            a2 = (const void*) ((uintptr_t) a2 - kc); // update 1st row matrix A pointer
            a3 = (const void*) ((uintptr_t) a3 - kc); // update 1st row matrix A pointer
        }
        nc -= vl;
    } while (nc != 0);
}

void xnn_f32_gemm_relu_ukernel_1x8__rvv_u2v(
        size_t mr,
        size_t nc,
        size_t kc,
        const float* restrict a,
        size_t a_stride,
        const float* restrict w,
        float* restrict c,
        size_t cm_stride,
        size_t cn_stride,
        const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(mr != 0);
    assert(mr <= 1);
    assert(nc != 0);
    assert(kc != 0);
    assert(kc % sizeof(float) == 0);
    assert(a != NULL);
    assert(w != NULL);
    assert(c != NULL);

    const float* a0 = a;
    float* c0 = c;

    size_t kcl = kc / sizeof(float);
    do {
        size_t vl = vsetvl_e32m2(nc);
        vfloat32m2_t vacc = vle32_v_f32m2(w, 8);
        w += 8;
        for(size_t k = 0; k < kcl ; k++){
            vfloat32m2_t vw = vle32_v_f32m2(w, 8);
            w += 8;
            vacc = vfmacc_vf_f32m2(vacc, *a0, vw, 8);
            a0++;
        }
        vacc = vfmax_vf_f32m2(vacc, 0.0f, vl);
        vse32_v_f32m2(c0, vacc, vl);
        if(nc >= 4){
            c0 = (float*) ((uintptr_t) c0 + cn_stride);
            a0 = (const void*) ((uintptr_t) a0 - kc);
        }
        nc -= vl;
    } while (nc != 0);
}

void xnn_f32_gemm_relu_ukernel_4x8__rvv_u2v(
        size_t mr,
        size_t nc,
        size_t kc,
        const float* restrict a,
        size_t a_stride,
        const float* restrict w,
        float* restrict c,
        size_t cm_stride,
        size_t cn_stride,
        const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(mr != 0);
    assert(mr <= 4); // max process 1 row
    assert(nc != 0);
    assert(kc != 0);
    assert(kc % sizeof(float) == 0);
    assert(a != NULL);
    assert(w != NULL);
    assert(c != NULL);

    const float* a0 = a;
    float* c0 = c;
    const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
    float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
    if XNN_UNPREDICTABLE(mr < 2) {
        a1 = a0;
        c1 = c0;
    }
    const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
    float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
    if XNN_UNPREDICTABLE(mr <= 2) {
        a2 = a1;
        c2 = c1;
    }
    const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
    float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
    if XNN_UNPREDICTABLE(mr != 4) {
        a3 = a2;
        c3 = c2;
    }

    size_t kcl = kc / sizeof(float);

    do {
        size_t vl = vsetvl_e32m2(nc); // vector length
        vfloat32m2_t vacc0 = vle32_v_f32m2(w, 8); // 1st row count
        vfloat32m2_t vacc1 = vle32_v_f32m2(w, 8); // 1st row count
        vfloat32m2_t vacc2 = vle32_v_f32m2(w, 8); // 1st row count
        vfloat32m2_t vacc3 = vle32_v_f32m2(w, 8); // 1st row count
        w += 8;
        for(size_t k = 0; k < kcl ; k++){
            vfloat32m2_t vw = vle32_v_f32m2(w, 8);
            w += 8;
            vacc0 = vfmacc_vf_f32m2(vacc0, *a0, vw, 8); // update 1st row count
            vacc1 = vfmacc_vf_f32m2(vacc1, *a1, vw, 8); // update 1st row count
            vacc2 = vfmacc_vf_f32m2(vacc2, *a2, vw, 8); // update 1st row count
            vacc3 = vfmacc_vf_f32m2(vacc3, *a3, vw, 8); // update 1st row count
            a0++;
            a1++;
            a2++;
            a3++;
        }
        vacc0 = vfmax_vf_f32m2(vacc0, 0.0f, vl);
        vacc1 = vfmax_vf_f32m2(vacc1, 0.0f, vl);
        vacc2 = vfmax_vf_f32m2(vacc2, 0.0f, vl);
        vacc3 = vfmax_vf_f32m2(vacc3, 0.0f, vl);

        vse32_v_f32m2(c0, vacc0, vl); // store 1st row result
        vse32_v_f32m2(c1, vacc1, vl); // store 1st row result
        vse32_v_f32m2(c2, vacc2, vl); // store 1st row result
        vse32_v_f32m2(c3, vacc3, vl); // store 1st row result
        if(nc >= 4){
            c0 = (float*) ((uintptr_t) c0 + cn_stride); // update 1st row matrix C pointer
            c1 = (float*) ((uintptr_t) c1 + cn_stride); // update 1st row matrix C pointer
            c2 = (float*) ((uintptr_t) c2 + cn_stride); // update 1st row matrix C pointer
            c3 = (float*) ((uintptr_t) c3 + cn_stride); // update 1st row matrix C pointer
            a0 = (const void*) ((uintptr_t) a0 - kc); // update 1st row matrix A pointer
            a1 = (const void*) ((uintptr_t) a1 - kc); // update 1st row matrix A pointer
            a2 = (const void*) ((uintptr_t) a2 - kc); // update 1st row matrix A pointer
            a3 = (const void*) ((uintptr_t) a3 - kc); // update 1st row matrix A pointer
        }
        nc -= vl;
    } while (nc != 0);
}


void xnn_f32_gemm_relu_ukernel_1x16__rvv_u4v(
        size_t mr,
        size_t nc,
        size_t kc,
        const float* restrict a,
        size_t a_stride,
        const float* restrict w,
        float* restrict c,
        size_t cm_stride,
        size_t cn_stride,
        const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(mr != 0);
    assert(mr <= 1);
    assert(nc != 0);
    assert(kc != 0);
    assert(kc % sizeof(float) == 0);
    assert(a != NULL);
    assert(w != NULL);
    assert(c != NULL);

    const float* a0 = a;
    float* c0 = c;
    size_t kcl = kc / sizeof(float);

    do {
        size_t vl = vsetvl_e32m4(nc);
        vfloat32m4_t vacc = vle32_v_f32m4(w, 16);
        w += 16;
        for(size_t k = 0; k < kcl ; k++){
            vfloat32m4_t vw = vle32_v_f32m4(w, 16);
            w += 16;
            vacc = vfmacc_vf_f32m4(vacc, *a0, vw, 16);
            a0++;
        }
        vacc = vfmax_vf_f32m4(vacc, 0.0f, vl);
        vse32_v_f32m4(c0, vacc, vl);
        if(nc >= 4){
            c0 = (float*) ((uintptr_t) c0 + cn_stride);
            a0 = (const void*) ((uintptr_t) a0 - kc);
        }
        nc -= vl;
    } while (nc != 0);
}

void xnn_f32_gemm_relu_ukernel_4x16__rvv_u4v(
        size_t mr,
        size_t nc,
        size_t kc,
        const float* restrict a,
        size_t a_stride,
        const float* restrict w,
        float* restrict c,
        size_t cm_stride,
        size_t cn_stride,
        const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(mr != 0);
    assert(mr <= 4); // max process 1 row
    assert(nc != 0);
    assert(kc != 0);
    assert(kc % sizeof(float) == 0);
    assert(a != NULL);
    assert(w != NULL);
    assert(c != NULL);

    const float* a0 = a;
    float* c0 = c;
    const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
    float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
    if XNN_UNPREDICTABLE(mr < 2) {
        a1 = a0;
        c1 = c0;
    }
    const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
    float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
    if XNN_UNPREDICTABLE(mr <= 2) {
        a2 = a1;
        c2 = c1;
    }
    const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
    float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
    if XNN_UNPREDICTABLE(mr != 4) {
        a3 = a2;
        c3 = c2;
    }

    size_t kcl = kc / sizeof(float);

    do {
        size_t vl = vsetvl_e32m4(nc); // vector length
        vfloat32m4_t vacc0 = vle32_v_f32m4(w, 16); // 1st row count
        vfloat32m4_t vacc1 = vle32_v_f32m4(w, 16); // 1st row count
        vfloat32m4_t vacc2 = vle32_v_f32m4(w, 16); // 1st row count
        vfloat32m4_t vacc3 = vle32_v_f32m4(w, 16); // 1st row count
        w += 16;
        for(size_t k = 0; k < kcl ; k++){
            vfloat32m4_t vw = vle32_v_f32m4(w, 16);
            w += 16;
            vacc0 = vfmacc_vf_f32m4(vacc0, *a0, vw, 16); // update 1st row count
            vacc1 = vfmacc_vf_f32m4(vacc1, *a1, vw, 16); // update 1st row count
            vacc2 = vfmacc_vf_f32m4(vacc2, *a2, vw, 16); // update 1st row count
            vacc3 = vfmacc_vf_f32m4(vacc3, *a3, vw, 16); // update 1st row count
            a0++;
            a1++;
            a2++;
            a3++;
        }
        vacc0 = vfmax_vf_f32m4(vacc0, 0.0f, vl);
        vacc1 = vfmax_vf_f32m4(vacc1, 0.0f, vl);
        vacc2 = vfmax_vf_f32m4(vacc2, 0.0f, vl);
        vacc3 = vfmax_vf_f32m4(vacc3, 0.0f, vl);
        vse32_v_f32m4(c0, vacc0, vl); // store 1st row result
        vse32_v_f32m4(c1, vacc1, vl); // store 1st row result
        vse32_v_f32m4(c2, vacc2, vl); // store 1st row result
        vse32_v_f32m4(c3, vacc3, vl); // store 1st row result
        if(nc >= 4){
            c0 = (float*) ((uintptr_t) c0 + cn_stride); // update 1st row matrix C pointer
            c1 = (float*) ((uintptr_t) c1 + cn_stride); // update 1st row matrix C pointer
            c2 = (float*) ((uintptr_t) c2 + cn_stride); // update 1st row matrix C pointer
            c3 = (float*) ((uintptr_t) c3 + cn_stride); // update 1st row matrix C pointer
            a0 = (const void*) ((uintptr_t) a0 - kc); // update 1st row matrix A pointer
            a1 = (const void*) ((uintptr_t) a1 - kc); // update 1st row matrix A pointer
            a2 = (const void*) ((uintptr_t) a2 - kc); // update 1st row matrix A pointer
            a3 = (const void*) ((uintptr_t) a3 - kc); // update 1st row matrix A pointer
        }
        nc -= vl;
    } while (nc != 0);
}

void xnn_f32_gemm_relu_ukernel_1x32__rvv_u8v(
        size_t mr,
        size_t nc,
        size_t kc,
        const float* restrict a,
        size_t a_stride,
        const float* restrict w,
        float* restrict c,
        size_t cm_stride,
        size_t cn_stride,
        const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
     assert(mr != 0);
    assert(mr <= 1);
    assert(nc != 0);
    assert(kc != 0);
    assert(kc % sizeof(float) == 0);
    assert(a != NULL);
    assert(w != NULL);
    assert(c != NULL);

    const float* a0 = a;
    float* c0 = c;
    size_t kcl = kc / sizeof(float);

    do {
        size_t vl = vsetvl_e32m8(nc);
        vfloat32m8_t vacc = vle32_v_f32m8(w, 32);
        w += 32;
        for(size_t k = 0; k < kcl ; k++){
            vfloat32m8_t vw = vle32_v_f32m8(w, 32);
            w += 32;
            vacc = vfmacc_vf_f32m8(vacc, *a0, vw, 32);
            a0++;
        }
        vacc = vfmax_vf_f32m8(vacc, 0.0f, vl);
        vse32_v_f32m8(c0, vacc, vl);
        if(nc >= 4){
            c0 = (float*) ((uintptr_t) c0 + cn_stride);
            a0 = (const void*) ((uintptr_t) a0 - kc);
        }
        nc -= vl;
    } while (nc != 0);
}

void xnn_f32_gemm_relu_ukernel_4x32__rvv_u8v(
    size_t mr,
        size_t nc,
        size_t kc,
        const float* restrict a,
        size_t a_stride,
        const float* restrict w,
        float* restrict c,
        size_t cm_stride,
        size_t cn_stride,
        const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    const float* a0 = a;
    float* c0 = c;
    const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
    float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
    if XNN_UNPREDICTABLE(mr < 2) {
        a1 = a0;
        c1 = c0;
    }
    const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
    float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
    if XNN_UNPREDICTABLE(mr <= 2) {
        a2 = a1;
        c2 = c1;
    }
    const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
    float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
    if XNN_UNPREDICTABLE(mr != 4) {
        a3 = a2;
        c3 = c2;
    }

    size_t kcl = kc / sizeof(float);

    do
    {
        size_t vl = vsetvl_e32m8(nc);
        vfloat32m8_t vacc0 = vle32_v_f32m8(w, 32); // 1st row count
        vfloat32m8_t vacc1 = vle32_v_f32m8(w, 32); // 1st row count
        vfloat32m8_t vacc2 = vle32_v_f32m8(w, 32); // 1st row count
        vfloat32m8_t vacc3 = vle32_v_f32m8(w, 32); // 1st row count
        w += 32;
        for(size_t k = 0; k < kcl ; k++){
            vfloat32m8_t vw = vle32_v_f32m8(w, 32);
            w += 32;
            vacc0 = vfmacc_vf_f32m8(vacc0, *a0, vw, 32); // update 1st row count
            vacc1 = vfmacc_vf_f32m8(vacc1, *a1, vw, 32); // update 1st row count
            vacc2 = vfmacc_vf_f32m8(vacc2, *a2, vw, 32); // update 1st row count
            vacc3 = vfmacc_vf_f32m8(vacc3, *a3, vw, 32); // update 1st row count
            a0++;
            a1++;
            a2++;
            a3++;
        }
        vacc0 = vfmax_vf_f32m8(vacc0, 0.0f, vl);
        vacc1 = vfmax_vf_f32m8(vacc1, 0.0f, vl);
        vacc2 = vfmax_vf_f32m8(vacc2, 0.0f, vl);
        vacc3 = vfmax_vf_f32m8(vacc3, 0.0f, vl);
        vse32_v_f32m8(c0, vacc0, vl); // store 1st row result
        vse32_v_f32m8(c1, vacc1, vl); // store 1st row result
        vse32_v_f32m8(c2, vacc2, vl); // store 1st row result
        vse32_v_f32m8(c3, vacc3, vl); // store 1st row result
        if(nc >= 4){
            c0 = (float*) ((uintptr_t) c0 + cn_stride); // update 1st row matrix C pointer
            c1 = (float*) ((uintptr_t) c1 + cn_stride); // update 1st row matrix C pointer
            c2 = (float*) ((uintptr_t) c2 + cn_stride); // update 1st row matrix C pointer
            c3 = (float*) ((uintptr_t) c3 + cn_stride); // update 1st row matrix C pointer
            a0 = (const void*) ((uintptr_t) a0 - kc); // update 1st row matrix A pointer
            a1 = (const void*) ((uintptr_t) a1 - kc); // update 1st row matrix A pointer
            a2 = (const void*) ((uintptr_t) a2 - kc); // update 1st row matrix A pointer
            a3 = (const void*) ((uintptr_t) a3 - kc); // update 1st row matrix A pointer
        }
        nc -= vl;
    } while (nc != 0);
    
}

void xnn_f32_gemm_minmax_ukernel_1x4__rvv_u1v(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
	assert(mr != 0);
	assert(mr <= 1);
	assert(nc != 0);
	assert(kc != 0);
	assert(kc % sizeof(float) == 0);
	assert(a != NULL);
	assert(w != NULL);
	assert(c != NULL);

	const float* a0 = a;
	float* c0 = c;
	size_t kcl = kc / sizeof(float);
    const float vmin = params->scalar.min;
    const float vmax = params->scalar.max;

	do {
		size_t vl = vsetvl_e32m1(nc);
		vfloat32m1_t vacc = vle32_v_f32m1(w, 4);
		w += 4;
		for(size_t k = 0; k < kcl ; k++){
			vfloat32m1_t vw = vle32_v_f32m1(w, 4);
			w += 4;
			vacc = vfmacc_vf_f32m1(vacc, *a0, vw, 4);
			a0++;
		}
        vacc = vfmax_vf_f32m1(vacc, vmin, vl);
        vacc = vfmin_vf_f32m1(vacc, vmax, vl);
		vse32_v_f32m1(c0, vacc, vl);
		if(nc >= 4){
      		c0 = (float*) ((uintptr_t) c0 + cn_stride);
      		a0 = (const void*) ((uintptr_t) a0 - kc);
		}
		nc -= vl;
	} while (nc != 0);
}

void xnn_f32_gemm_minmax_ukernel_4x4__rvv_u1v(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(mr != 0);
    assert(mr <= 4); // max process 1 row
    assert(nc != 0);
    assert(kc != 0);
    assert(kc % sizeof(float) == 0);
    assert(a != NULL);
    assert(w != NULL);
    assert(c != NULL);

    const float* a0 = a;
    float* c0 = c;
    const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
    float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
    if XNN_UNPREDICTABLE(mr < 2) {
        a1 = a0;
        c1 = c0;
    }
    const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
    float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
    if XNN_UNPREDICTABLE(mr <= 2) {
        a2 = a1;
        c2 = c1;
    }
    const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
    float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
    if XNN_UNPREDICTABLE(mr != 4) {
        a3 = a2;
        c3 = c2;
    }
    const float vmin = params->scalar.min;
    const float vmax = params->scalar.max;

    size_t kcl = kc / sizeof(float);

    do {
        size_t vl = vsetvl_e32m1(nc); // vector length
        vfloat32m1_t vacc0 = vle32_v_f32m1(w, 4); // 1st row count
        vfloat32m1_t vacc1 = vle32_v_f32m1(w, 4); // 1st row count
        vfloat32m1_t vacc2 = vle32_v_f32m1(w, 4); // 1st row count
        vfloat32m1_t vacc3 = vle32_v_f32m1(w, 4); // 1st row count
        w += 4;
        for(size_t k = 0; k < kcl ; k++){
            vfloat32m1_t vw = vle32_v_f32m1(w, 4);
            w += 4;
            vacc0 = vfmacc_vf_f32m1(vacc0, *a0, vw, 4); // update 1st row count
            vacc1 = vfmacc_vf_f32m1(vacc1, *a1, vw, 4); // update 1st row count
            vacc2 = vfmacc_vf_f32m1(vacc2, *a2, vw, 4); // update 1st row count
            vacc3 = vfmacc_vf_f32m1(vacc3, *a3, vw, 4); // update 1st row count
            a0++;
            a1++;
            a2++;
            a3++;
        }
        vacc0 = vfmax_vf_f32m1(vacc0, vmin, vl);
        vacc1 = vfmax_vf_f32m1(vacc1, vmin, vl);
        vacc2 = vfmax_vf_f32m1(vacc2, vmin, vl);
        vacc3 = vfmax_vf_f32m1(vacc3, vmin, vl);

        vacc0 = vfmin_vf_f32m1(vacc0, vmax, vl);
        vacc1 = vfmin_vf_f32m1(vacc1, vmax, vl);
        vacc2 = vfmin_vf_f32m1(vacc2, vmax, vl);
        vacc3 = vfmin_vf_f32m1(vacc3, vmax, vl);
        vse32_v_f32m1(c0, vacc0, vl); // store 1st row result
        vse32_v_f32m1(c1, vacc1, vl); // store 1st row result
        vse32_v_f32m1(c2, vacc2, vl); // store 1st row result
        vse32_v_f32m1(c3, vacc3, vl); // store 1st row result
        if(nc >= 4){
            c0 = (float*) ((uintptr_t) c0 + cn_stride); // update 1st row matrix C pointer
            c1 = (float*) ((uintptr_t) c1 + cn_stride); // update 1st row matrix C pointer
            c2 = (float*) ((uintptr_t) c2 + cn_stride); // update 1st row matrix C pointer
            c3 = (float*) ((uintptr_t) c3 + cn_stride); // update 1st row matrix C pointer
            a0 = (const void*) ((uintptr_t) a0 - kc); // update 1st row matrix A pointer
            a1 = (const void*) ((uintptr_t) a1 - kc); // update 1st row matrix A pointer
            a2 = (const void*) ((uintptr_t) a2 - kc); // update 1st row matrix A pointer
            a3 = (const void*) ((uintptr_t) a3 - kc); // update 1st row matrix A pointer
        }
        nc -= vl;
    } while (nc != 0);
}

void xnn_f32_gemm_minmax_ukernel_1x8__rvv_u2v(
        size_t mr,
        size_t nc,
        size_t kc,
        const float* restrict a,
        size_t a_stride,
        const float* restrict w,
        float* restrict c,
        size_t cm_stride,
        size_t cn_stride,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(mr != 0);
    assert(mr <= 1);
    assert(nc != 0);
    assert(kc != 0);
    assert(kc % sizeof(float) == 0);
    assert(a != NULL);
    assert(w != NULL);
    assert(c != NULL);

    const float* a0 = a;
    float* c0 = c;

    const float vmin = params->scalar.min;
    const float vmax = params->scalar.max;

    size_t kcl = kc / sizeof(float);
    do {
        size_t vl = vsetvl_e32m2(nc);
        vfloat32m2_t vacc = vle32_v_f32m2(w, 8);
        w += 8;
        for(size_t k = 0; k < kcl ; k++){
            vfloat32m2_t vw = vle32_v_f32m2(w, 8);
            w += 8;
            vacc = vfmacc_vf_f32m2(vacc, *a0, vw, 8);
            a0++;
        }
        vacc = vfmax_vf_f32m2(vacc, vmin, vl);
        vacc = vfmin_vf_f32m2(vacc, vmax, vl);
        vse32_v_f32m2(c0, vacc, vl);
        if(nc >= 4){
            c0 = (float*) ((uintptr_t) c0 + cn_stride);
            a0 = (const void*) ((uintptr_t) a0 - kc);
        }
        nc -= vl;
    } while (nc != 0);
}

void xnn_f32_gemm_minmax_ukernel_4x8__rvv_u2v(
        size_t mr,
        size_t nc,
        size_t kc,
        const float* restrict a,
        size_t a_stride,
        const float* restrict w,
        float* restrict c,
        size_t cm_stride,
        size_t cn_stride,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(mr != 0);
    assert(mr <= 4); // max process 1 row
    assert(nc != 0);
    assert(kc != 0);
    assert(kc % sizeof(float) == 0);
    assert(a != NULL);
    assert(w != NULL);
    assert(c != NULL);

    const float* a0 = a;
    float* c0 = c;
    const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
    float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
    if XNN_UNPREDICTABLE(mr < 2) {
        a1 = a0;
        c1 = c0;
    }
    const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
    float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
    if XNN_UNPREDICTABLE(mr <= 2) {
        a2 = a1;
        c2 = c1;
    }
    const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
    float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
    if XNN_UNPREDICTABLE(mr != 4) {
        a3 = a2;
        c3 = c2;
    }

    const float vmin = params->scalar.min;
    const float vmax = params->scalar.max;

    size_t kcl = kc / sizeof(float);

    do {
        size_t vl = vsetvl_e32m2(nc); // vector length
        vfloat32m2_t vacc0 = vle32_v_f32m2(w, 8); // 1st row count
        vfloat32m2_t vacc1 = vle32_v_f32m2(w, 8); // 1st row count
        vfloat32m2_t vacc2 = vle32_v_f32m2(w, 8); // 1st row count
        vfloat32m2_t vacc3 = vle32_v_f32m2(w, 8); // 1st row count
        w += 8;
        for(size_t k = 0; k < kcl ; k++){
            vfloat32m2_t vw = vle32_v_f32m2(w, 8);
            w += 8;
            vacc0 = vfmacc_vf_f32m2(vacc0, *a0, vw, 8); // update 1st row count
            vacc1 = vfmacc_vf_f32m2(vacc1, *a1, vw, 8); // update 1st row count
            vacc2 = vfmacc_vf_f32m2(vacc2, *a2, vw, 8); // update 1st row count
            vacc3 = vfmacc_vf_f32m2(vacc3, *a3, vw, 8); // update 1st row count
            a0++;
            a1++;
            a2++;
            a3++;
        }
        vacc0 = vfmax_vf_f32m2(vacc0, vmin, vl);
        vacc1 = vfmax_vf_f32m2(vacc1, vmin, vl);
        vacc2 = vfmax_vf_f32m2(vacc2, vmin, vl);
        vacc3 = vfmax_vf_f32m2(vacc3, vmin, vl);

        vacc0 = vfmin_vf_f32m2(vacc0, vmax, vl);
        vacc1 = vfmin_vf_f32m2(vacc1, vmax, vl);
        vacc2 = vfmin_vf_f32m2(vacc2, vmax, vl);
        vacc3 = vfmin_vf_f32m2(vacc3, vmax, vl);

        vse32_v_f32m2(c0, vacc0, vl); // store 1st row result
        vse32_v_f32m2(c1, vacc1, vl); // store 1st row result
        vse32_v_f32m2(c2, vacc2, vl); // store 1st row result
        vse32_v_f32m2(c3, vacc3, vl); // store 1st row result
        if(nc >= 4){
            c0 = (float*) ((uintptr_t) c0 + cn_stride); // update 1st row matrix C pointer
            c1 = (float*) ((uintptr_t) c1 + cn_stride); // update 1st row matrix C pointer
            c2 = (float*) ((uintptr_t) c2 + cn_stride); // update 1st row matrix C pointer
            c3 = (float*) ((uintptr_t) c3 + cn_stride); // update 1st row matrix C pointer
            a0 = (const void*) ((uintptr_t) a0 - kc); // update 1st row matrix A pointer
            a1 = (const void*) ((uintptr_t) a1 - kc); // update 1st row matrix A pointer
            a2 = (const void*) ((uintptr_t) a2 - kc); // update 1st row matrix A pointer
            a3 = (const void*) ((uintptr_t) a3 - kc); // update 1st row matrix A pointer
        }
        nc -= vl;
    } while (nc != 0);
}


void xnn_f32_gemm_minmax_ukernel_1x16__rvv_u4v(
        size_t mr,
        size_t nc,
        size_t kc,
        const float* restrict a,
        size_t a_stride,
        const float* restrict w,
        float* restrict c,
        size_t cm_stride,
        size_t cn_stride,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(mr != 0);
    assert(mr <= 1);
    assert(nc != 0);
    assert(kc != 0);
    assert(kc % sizeof(float) == 0);
    assert(a != NULL);
    assert(w != NULL);
    assert(c != NULL);

    const float* a0 = a;
    float* c0 = c;
    size_t kcl = kc / sizeof(float);

    const float vmin = params->scalar.min;
    const float vmax = params->scalar.max;

    do {
        size_t vl = vsetvl_e32m4(nc);
        vfloat32m4_t vacc = vle32_v_f32m4(w, 16);
        w += 16;
        for(size_t k = 0; k < kcl ; k++){
            vfloat32m4_t vw = vle32_v_f32m4(w, 16);
            w += 16;
            vacc = vfmacc_vf_f32m4(vacc, *a0, vw, 16);
            a0++;
        }
        vacc = vfmax_vf_f32m4(vacc, vmin, vl);
        vacc = vfmin_vf_f32m4(vacc, vmax, vl);
        vse32_v_f32m4(c0, vacc, vl);
        if(nc >= 4){
            c0 = (float*) ((uintptr_t) c0 + cn_stride);
            a0 = (const void*) ((uintptr_t) a0 - kc);
        }
        nc -= vl;
    } while (nc != 0);
}

void xnn_f32_gemm_minmax_ukernel_4x16__rvv_u4v(
        size_t mr,
        size_t nc,
        size_t kc,
        const float* restrict a,
        size_t a_stride,
        const float* restrict w,
        float* restrict c,
        size_t cm_stride,
        size_t cn_stride,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(mr != 0);
    assert(mr <= 4); // max process 1 row
    assert(nc != 0);
    assert(kc != 0);
    assert(kc % sizeof(float) == 0);
    assert(a != NULL);
    assert(w != NULL);
    assert(c != NULL);

    const float* a0 = a;
    float* c0 = c;
    const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
    float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
    if XNN_UNPREDICTABLE(mr < 2) {
        a1 = a0;
        c1 = c0;
    }
    const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
    float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
    if XNN_UNPREDICTABLE(mr <= 2) {
        a2 = a1;
        c2 = c1;
    }
    const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
    float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
    if XNN_UNPREDICTABLE(mr != 4) {
        a3 = a2;
        c3 = c2;
    }

    const float vmin = params->scalar.min;
    const float vmax = params->scalar.max;

    size_t kcl = kc / sizeof(float);

    do {
        size_t vl = vsetvl_e32m4(nc); // vector length
        vfloat32m4_t vacc0 = vle32_v_f32m4(w, 16); // 1st row count
        vfloat32m4_t vacc1 = vle32_v_f32m4(w, 16); // 1st row count
        vfloat32m4_t vacc2 = vle32_v_f32m4(w, 16); // 1st row count
        vfloat32m4_t vacc3 = vle32_v_f32m4(w, 16); // 1st row count
        w += 16;
        for(size_t k = 0; k < kcl ; k++){
            vfloat32m4_t vw = vle32_v_f32m4(w, 16);
            w += 16;
            vacc0 = vfmacc_vf_f32m4(vacc0, *a0, vw, 16); // update 1st row count
            vacc1 = vfmacc_vf_f32m4(vacc1, *a1, vw, 16); // update 1st row count
            vacc2 = vfmacc_vf_f32m4(vacc2, *a2, vw, 16); // update 1st row count
            vacc3 = vfmacc_vf_f32m4(vacc3, *a3, vw, 16); // update 1st row count
            a0++;
            a1++;
            a2++;
            a3++;
        }
        vacc0 = vfmax_vf_f32m4(vacc0, vmin, vl);
        vacc1 = vfmax_vf_f32m4(vacc1, vmin, vl);
        vacc2 = vfmax_vf_f32m4(vacc2, vmin, vl);
        vacc3 = vfmax_vf_f32m4(vacc3, vmin, vl);

        vacc0 = vfmin_vf_f32m4(vacc0, vmax, vl);
        vacc1 = vfmin_vf_f32m4(vacc1, vmax, vl);
        vacc2 = vfmin_vf_f32m4(vacc2, vmax, vl);
        vacc3 = vfmin_vf_f32m4(vacc3, vmax, vl);
        vse32_v_f32m4(c0, vacc0, vl); // store 1st row result
        vse32_v_f32m4(c1, vacc1, vl); // store 1st row result
        vse32_v_f32m4(c2, vacc2, vl); // store 1st row result
        vse32_v_f32m4(c3, vacc3, vl); // store 1st row result
        if(nc >= 4){
            c0 = (float*) ((uintptr_t) c0 + cn_stride); // update 1st row matrix C pointer
            c1 = (float*) ((uintptr_t) c1 + cn_stride); // update 1st row matrix C pointer
            c2 = (float*) ((uintptr_t) c2 + cn_stride); // update 1st row matrix C pointer
            c3 = (float*) ((uintptr_t) c3 + cn_stride); // update 1st row matrix C pointer
            a0 = (const void*) ((uintptr_t) a0 - kc); // update 1st row matrix A pointer
            a1 = (const void*) ((uintptr_t) a1 - kc); // update 1st row matrix A pointer
            a2 = (const void*) ((uintptr_t) a2 - kc); // update 1st row matrix A pointer
            a3 = (const void*) ((uintptr_t) a3 - kc); // update 1st row matrix A pointer
        }
        nc -= vl;
    } while (nc != 0);
}

void xnn_f32_gemm_minmax_ukernel_1x32__rvv_u8v(
        size_t mr,
        size_t nc,
        size_t kc,
        const float* restrict a,
        size_t a_stride,
        const float* restrict w,
        float* restrict c,
        size_t cm_stride,
        size_t cn_stride,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
     assert(mr != 0);
    assert(mr <= 1);
    assert(nc != 0);
    assert(kc != 0);
    assert(kc % sizeof(float) == 0);
    assert(a != NULL);
    assert(w != NULL);
    assert(c != NULL);

    const float* a0 = a;
    float* c0 = c;
    size_t kcl = kc / sizeof(float);
    const float vmin = params->scalar.min;
    const float vmax = params->scalar.max;

    do {
        size_t vl = vsetvl_e32m8(nc);
        vfloat32m8_t vacc = vle32_v_f32m8(w, 32);
        w += 32;
        for(size_t k = 0; k < kcl ; k++){
            vfloat32m8_t vw = vle32_v_f32m8(w, 32);
            w += 32;
            vacc = vfmacc_vf_f32m8(vacc, *a0, vw, 32);
            a0++;
        }
        vacc = vfmax_vf_f32m8(vacc, vmin, vl);
        vacc = vfmin_vf_f32m8(vacc, vmax, vl);
        vse32_v_f32m8(c0, vacc, vl);
        if(nc >= 4){
            c0 = (float*) ((uintptr_t) c0 + cn_stride);
            a0 = (const void*) ((uintptr_t) a0 - kc);
        }
        nc -= vl;
    } while (nc != 0);
}

void xnn_f32_gemm_minmax_ukernel_4x32__rvv_u8v(
    size_t mr,
        size_t nc,
        size_t kc,
        const float* restrict a,
        size_t a_stride,
        const float* restrict w,
        float* restrict c,
        size_t cm_stride,
        size_t cn_stride,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    const float* a0 = a;
    float* c0 = c;
    const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
    float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
    if XNN_UNPREDICTABLE(mr < 2) {
        a1 = a0;
        c1 = c0;
    }
    const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
    float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
    if XNN_UNPREDICTABLE(mr <= 2) {
        a2 = a1;
        c2 = c1;
    }
    const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
    float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
    if XNN_UNPREDICTABLE(mr != 4) {
        a3 = a2;
        c3 = c2;
    }

    size_t kcl = kc / sizeof(float);
    const float vmin = params->scalar.min;
    const float vmax = params->scalar.max;

    do
    {
        size_t vl = vsetvl_e32m8(nc);
        vfloat32m8_t vacc0 = vle32_v_f32m8(w, 32); // 1st row count
        vfloat32m8_t vacc1 = vle32_v_f32m8(w, 32); // 1st row count
        vfloat32m8_t vacc2 = vle32_v_f32m8(w, 32); // 1st row count
        vfloat32m8_t vacc3 = vle32_v_f32m8(w, 32); // 1st row count
        w += 32;
        for(size_t k = 0; k < kcl ; k++){
            vfloat32m8_t vw = vle32_v_f32m8(w, 32);
            w += 32;
            vacc0 = vfmacc_vf_f32m8(vacc0, *a0, vw, 32); // update 1st row count
            vacc1 = vfmacc_vf_f32m8(vacc1, *a1, vw, 32); // update 1st row count
            vacc2 = vfmacc_vf_f32m8(vacc2, *a2, vw, 32); // update 1st row count
            vacc3 = vfmacc_vf_f32m8(vacc3, *a3, vw, 32); // update 1st row count
            a0++;
            a1++;
            a2++;
            a3++;
        }
        vacc0 = vfmax_vf_f32m8(vacc0, vmin, vl);
        vacc1 = vfmax_vf_f32m8(vacc1, vmin, vl);
        vacc2 = vfmax_vf_f32m8(vacc2, vmin, vl);
        vacc3 = vfmax_vf_f32m8(vacc3, vmin, vl);

        vacc0 = vfmin_vf_f32m8(vacc0, vmax, vl);
        vacc1 = vfmin_vf_f32m8(vacc1, vmax, vl);
        vacc2 = vfmin_vf_f32m8(vacc2, vmax, vl);
        vacc3 = vfmin_vf_f32m8(vacc3, vmax, vl);
        vse32_v_f32m8(c0, vacc0, vl); // store 1st row result
        vse32_v_f32m8(c1, vacc1, vl); // store 1st row result
        vse32_v_f32m8(c2, vacc2, vl); // store 1st row result
        vse32_v_f32m8(c3, vacc3, vl); // store 1st row result
        if(nc >= 4){
            c0 = (float*) ((uintptr_t) c0 + cn_stride); // update 1st row matrix C pointer
            c1 = (float*) ((uintptr_t) c1 + cn_stride); // update 1st row matrix C pointer
            c2 = (float*) ((uintptr_t) c2 + cn_stride); // update 1st row matrix C pointer
            c3 = (float*) ((uintptr_t) c3 + cn_stride); // update 1st row matrix C pointer
            a0 = (const void*) ((uintptr_t) a0 - kc); // update 1st row matrix A pointer
            a1 = (const void*) ((uintptr_t) a1 - kc); // update 1st row matrix A pointer
            a2 = (const void*) ((uintptr_t) a2 - kc); // update 1st row matrix A pointer
            a3 = (const void*) ((uintptr_t) a3 - kc); // update 1st row matrix A pointer
        }
        nc -= vl;
    } while (nc != 0);
    
}


void xnn_f32_igemm_minmax_ukernel_1x4__rvv_u1v(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(mr != 0);
    assert(mr <= 1);
    assert(nc != 0);
    assert(kc != 0);
    assert(kc % sizeof(float) == 0);
    assert(ks != 0);
    assert(ks % (1 * sizeof(void*)) == 0);
    assert(a_offset % sizeof(float) == 0);
    assert(a != NULL);
    assert(w != NULL);
    assert(c != NULL);

    float* c0 = c;

    const float vmin = params->scalar.min;
    const float vmax = params->scalar.max;

    do {
        size_t vl = vsetvl_e32m1(nc); // vector length
        vfloat32m1_t vacc0 = vle32_v_f32m1(w, 4); // 1st row count
        w += 4;

         size_t p = ks;
        size_t kcl = kc / sizeof(float);
        do {
             const float* restrict a0 = a[0];
            assert(a0 != NULL);
            if XNN_UNPREDICTABLE(a0 != zero) {
                a0 = (const float*) ((uintptr_t) a0 + a_offset);
            }
            a += 1;

            size_t k = kc;
            for(size_t k = 0; k < kcl ; k++){
                vfloat32m1_t vw = vle32_v_f32m1(w, 4);
                w += 4;
                vacc0 = vfmacc_vf_f32m1(vacc0, *a0, vw, 4); // update 1st row count
                a0++;
            }
            p -= 1 * sizeof(void*);
        } while (p != 0);
        vacc0 = vfmax_vf_f32m1(vacc0, vmin, vl);
        vacc0 = vfmin_vf_f32m1(vacc0, vmax, vl);
        vse32_v_f32m1(c0, vacc0, vl); // store 1st row result

        if XNN_LIKELY(nc >= 4) {
            c0 = (float*) ((uintptr_t) c0 + cn_stride);
            a = (const float**restrict) ((uintptr_t) a - ks);
        }
        nc -= vl;
    } while (nc != 0);
}

void xnn_f32_igemm_ukernel_4x4__rvv_u1v(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(mr != 0);
    assert(mr <= 4);
    assert(nc != 0);
    assert(kc != 0);
    assert(kc % sizeof(float) == 0);
    assert(ks != 0);
    assert(ks % (4 * sizeof(void*)) == 0);
    assert(a_offset % sizeof(float) == 0);
    assert(a != NULL);
    assert(w != NULL);
    assert(c != NULL);

    float* c0 = c;
    float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
    if XNN_UNPREDICTABLE(mr < 2) {
        c1 = c0;
    }
    float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
    if XNN_UNPREDICTABLE(mr <= 2) {
        c2 = c1;
    }
    float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
    if XNN_UNPREDICTABLE(mr != 4) {
        c3 = c2;
    }

    const float vmin = params->scalar.min;
    const float vmax = params->scalar.max;

    do {
        size_t vl = vsetvl_e32m1(nc); // vector length
        vfloat32m1_t vacc0 = vle32_v_f32m1(w, 4); // 1st row count
        vfloat32m1_t vacc1 = vle32_v_f32m1(w, 4); // 1st row count
        vfloat32m1_t vacc2 = vle32_v_f32m1(w, 4); // 1st row count
        vfloat32m1_t vacc3 = vle32_v_f32m1(w, 4); // 1st row count
        w += 4;

        size_t p = ks;
        size_t kcl = kc / sizeof(float);
        do {
            const float* restrict a0 = a[0];
            assert(a0 != NULL);
            if XNN_UNPREDICTABLE(a0 != zero) {
                a0 = (const float*) ((uintptr_t) a0 + a_offset);
            }
            const float* restrict a1 = a[1];
            assert(a1 != NULL);
            if XNN_UNPREDICTABLE(a1 != zero) {
                a1 = (const float*) ((uintptr_t) a1 + a_offset);
            }
            const float* restrict a2 = a[2];
            assert(a2 != NULL);
            if XNN_UNPREDICTABLE(a2 != zero) {
                a2 = (const float*) ((uintptr_t) a2 + a_offset);
            }
            const float* restrict a3 = a[3];
            assert(a3 != NULL);
            if XNN_UNPREDICTABLE(a3 != zero) {
                a3 = (const float*) ((uintptr_t) a3 + a_offset);
            }
            a += 4;

            size_t k = kc;
            for(size_t k = 0; k < kcl ; k++){
                vfloat32m1_t vw = vle32_v_f32m1(w, 4);
                w += 4;
                vacc0 = vfmacc_vf_f32m1(vacc0, *a0, vw, 4); // update 1st row count
                vacc1 = vfmacc_vf_f32m1(vacc1, *a1, vw, 4); // update 1st row count
                vacc2 = vfmacc_vf_f32m1(vacc2, *a2, vw, 4); // update 1st row count
                vacc3 = vfmacc_vf_f32m1(vacc3, *a3, vw, 4); // update 1st row count
                a0++;
                a1++;
                a2++;
                a3++;
            }
            p -= 4 * sizeof(void*);
        } while (p != 0);
        vacc0 = vfmax_vf_f32m1(vacc0, vmin, vl);
        vacc1 = vfmax_vf_f32m1(vacc1, vmin, vl);
        vacc2 = vfmax_vf_f32m1(vacc2, vmin, vl);
        vacc3 = vfmax_vf_f32m1(vacc3, vmin, vl);

        vacc0 = vfmin_vf_f32m1(vacc0, vmax, vl);
        vacc1 = vfmin_vf_f32m1(vacc1, vmax, vl);
        vacc2 = vfmin_vf_f32m1(vacc2, vmax, vl);
        vacc3 = vfmin_vf_f32m1(vacc3, vmax, vl);
        vse32_v_f32m1(c0, vacc0, vl); // store 1st row result
        vse32_v_f32m1(c1, vacc1, vl); // store 1st row result
        vse32_v_f32m1(c2, vacc2, vl); // store 1st row result
        vse32_v_f32m1(c3, vacc3, vl); // store 1st row result

        if XNN_LIKELY(nc >= 4) {
            c3 = (float*) ((uintptr_t) c3 + cn_stride);
            c2 = (float*) ((uintptr_t) c2 + cn_stride);
            c1 = (float*) ((uintptr_t) c1 + cn_stride);
            c0 = (float*) ((uintptr_t) c0 + cn_stride);

            a = (const float**restrict) ((uintptr_t) a - ks);
        }
        nc -= vl;
    } while (nc != 0);
}

void xnn_f32_igemm_minmax_ukernel_1x8__rvv_u2v(
        size_t mr,
        size_t nc,
        size_t kc,
        size_t ks,
        const float** restrict a,
        const float* restrict w,
        float* restrict c,
        size_t cm_stride,
        size_t cn_stride,
        size_t a_offset,
        const float* zero,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(mr != 0);
    assert(mr <= 1);
    assert(nc != 0);
    assert(kc != 0);
    assert(kc % sizeof(float) == 0);
    assert(ks != 0);
    assert(ks % (1 * sizeof(void*)) == 0);
    assert(a_offset % sizeof(float) == 0);
    assert(a != NULL);
    assert(w != NULL);
    assert(c != NULL);

    float* c0 = c;

    const float vmin = params->scalar.min;
    const float vmax = params->scalar.max;

    do {
        size_t vl = vsetvl_e32m2(nc); // vector length
        vfloat32m2_t vacc0 = vle32_v_f32m2(w, 8); // 1st row count
        w += 8;

        size_t p = ks;
        size_t kcl = kc / sizeof(float);
        do {
            const float* restrict a0 = a[0];
            assert(a0 != NULL);
            if XNN_UNPREDICTABLE(a0 != zero) {
                a0 = (const float*) ((uintptr_t) a0 + a_offset);
            }
            a += 1;

            size_t k = kc;
            for(size_t k = 0; k < kcl ; k++){
                vfloat32m2_t vw = vle32_v_f32m2(w, 8);
                w += 8;
                vacc0 = vfmacc_vf_f32m2(vacc0, *a0, vw, 8); // update 1st row count
                a0++;
            }
            p -= 1 * sizeof(void*);
        } while (p != 0);
        vacc0 = vfmax_vf_f32m2(vacc0, vmin, vl);
        vacc0 = vfmin_vf_f32m2(vacc0, vmax, vl);
        vse32_v_f32m2(c0, vacc0, vl); // store 1st row result

        if XNN_LIKELY(nc >= 4) {
            c0 = (float*) ((uintptr_t) c0 + cn_stride);
            a = (const float**restrict) ((uintptr_t) a - ks);
        }
        nc -= vl;
    } while (nc != 0);
}


void xnn_f32_igemm_minmax_ukernel_4x8__rvv_u2v(
        size_t mr,
        size_t nc,
        size_t kc,
        size_t ks,
        const float** restrict a,
        const float* restrict w,
        float* restrict c,
        size_t cm_stride,
        size_t cn_stride,
        size_t a_offset,const float* zero,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(mr != 0);
    assert(mr <= 4);
    assert(nc != 0);
    assert(kc != 0);
    assert(kc % sizeof(float) == 0);
    assert(ks != 0);
    assert(ks % (4 * sizeof(void*)) == 0);
    assert(a_offset % sizeof(float) == 0);
    assert(a != NULL);
    assert(w != NULL);
    assert(c != NULL);

    float* c0 = c;
    float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
    if XNN_UNPREDICTABLE(mr < 2) {
        c1 = c0;
    }
    float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
    if XNN_UNPREDICTABLE(mr <= 2) {
        c2 = c1;
    }
    float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
    if XNN_UNPREDICTABLE(mr != 4) {
        c3 = c2;
    }

    const float vmin = params->scalar.min;
    const float vmax = params->scalar.max;

    // print someting for debug


    do {
        size_t vl = vsetvl_e32m2(nc); // vector length
        vfloat32m2_t vacc0 = vle32_v_f32m2(w, 8); // 1st row count
        vfloat32m2_t vacc1 = vle32_v_f32m2(w, 8); // 1st row count
        vfloat32m2_t vacc2 = vle32_v_f32m2(w, 8); // 1st row count
        vfloat32m2_t vacc3 = vle32_v_f32m2(w, 8); // 1st row count
        w += 8;

        size_t p = ks;
        size_t kcl = kc / sizeof(float);
        do {
            const float* restrict a0 = a[0];
            assert(a0 != NULL);
            if XNN_UNPREDICTABLE(a0 != zero) {
                a0 = (const float*) ((uintptr_t) a0 + a_offset);
            }
            const float* restrict a1 = a[1];
            assert(a1 != NULL);
            if XNN_UNPREDICTABLE(a1 != zero) {
                a1 = (const float*) ((uintptr_t) a1 + a_offset);
            }
            const float* restrict a2 = a[2];
            assert(a2 != NULL);
            if XNN_UNPREDICTABLE(a2 != zero) {
                a2 = (const float*) ((uintptr_t) a2 + a_offset);
            }
            const float* restrict a3 = a[3];
            assert(a3 != NULL);
            if XNN_UNPREDICTABLE(a3 != zero) {
                a3 = (const float*) ((uintptr_t) a3 + a_offset);
            }
            a += 4;

            size_t k = kc;
            for(size_t k = 0; k < kcl ; k++){
                vfloat32m2_t vw = vle32_v_f32m2(w, 8);
                w += 8;
                vacc0 = vfmacc_vf_f32m2(vacc0, *a0, vw, 8); // update 1st row count
                vacc1 = vfmacc_vf_f32m2(vacc1, *a1, vw, 8); // update 1st row count
                vacc2 = vfmacc_vf_f32m2(vacc2, *a2, vw, 8); // update 1st row count
                vacc3 = vfmacc_vf_f32m2(vacc3, *a3, vw, 8); // update 1st row count
                a0++;
                a1++;
                a2++;
                a3++;
            }
            p -= 4 * sizeof(void*);
        } while (p != 0);
        vacc0 = vfmax_vf_f32m2(vacc0, vmin, vl);
        vacc1 = vfmax_vf_f32m2(vacc1, vmin, vl);
        vacc2 = vfmax_vf_f32m2(vacc2, vmin, vl);
        vacc3 = vfmax_vf_f32m2(vacc3, vmin, vl);

        vacc0 = vfmin_vf_f32m2(vacc0, vmax, vl);
        vacc1 = vfmin_vf_f32m2(vacc1, vmax, vl);
        vacc2 = vfmin_vf_f32m2(vacc2, vmax, vl);
        vacc3 = vfmin_vf_f32m2(vacc3, vmax, vl);

        vse32_v_f32m2(c0, vacc0, vl); // store 1st row result
        vse32_v_f32m2(c1, vacc1, vl); // store 1st row result
        vse32_v_f32m2(c2, vacc2, vl); // store 1st row result
        vse32_v_f32m2(c3, vacc3, vl); // store 1st row result

        if XNN_LIKELY(nc >= 4) {
            c3 = (float*) ((uintptr_t) c3 + cn_stride);
            c2 = (float*) ((uintptr_t) c2 + cn_stride);
            c1 = (float*) ((uintptr_t) c1 + cn_stride);
            c0 = (float*) ((uintptr_t) c0 + cn_stride);

            a = (const float**restrict) ((uintptr_t) a - ks);
        }
        nc -= vl;
    } while (nc != 0);
}

void xnn_f32_igemm_minmax_ukernel_1x16__rvv_u2v(
        size_t mr,
        size_t nc,
        size_t kc,
        size_t ks,
        const float** restrict a,
        const float* restrict w,
        float* restrict c,
        size_t cm_stride,
        size_t cn_stride,
        size_t a_offset,
        const float* zero,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(mr != 0);
    assert(mr <= 1);
    assert(nc != 0);
    assert(kc != 0);
    assert(kc % sizeof(float) == 0);
    assert(ks != 0);
    assert(ks % (1 * sizeof(void*)) == 0);
    assert(a_offset % sizeof(float) == 0);
    assert(a!= NULL);
    assert(w!= NULL);
    assert(c!= NULL);

    float* c0 = c;

    const float vmin = params->scalar.min;
    const float vmax = params->scalar.max;

    do {
        size_t vl = vsetvl_e32m4(nc); // vector length
        vfloat32m4_t vacc0 = vle32_v_f32m4(w, 16); // 1st row count
        w += 16;

        size_t p = ks;
        size_t kcl = kc / sizeof(float);
        do {
            const float* restrict a0 = a[0];
            assert(a0 != NULL);
            if XNN_UNPREDICTABLE(a0 != zero) {
                a0 = (const float*) ((uintptr_t) a0 + a_offset);
            }
            a += 1;

            size_t k = kc;
            for(size_t k = 0; k < kcl ; k++){
                vfloat32m4_t vw = vle32_v_f32m4(w, 16);
                w += 16;
                vacc0 = vfmacc_vf_f32m4(vacc0, *a0, vw, 16); // update 1st row count
                a0++;
            }
            p -= 1 * sizeof(void*);
        } while (p != 0);
        vacc0 = vfmax_vf_f32m4(vacc0, vmin, vl);
        vacc0 = vfmin_vf_f32m4(vacc0, vmax, vl);
        vse32_v_f32m4(c0, vacc0, vl); // store 1st row result

        if XNN_LIKELY(nc >= 4) {
            c0 = (float*) ((uintptr_t) c0 + cn_stride);
            a = (const float**restrict) ((uintptr_t) a - ks);
        }
        nc -= vl;
    } while (nc != 0);
}

void xnn_f32_igemm_ukernel_4x16__rvv_u4v(
        size_t mr,
        size_t nc,
        size_t kc,
        size_t ks,
        const float** restrict a,
        const float* restrict w,
        float* restrict c,
        size_t cm_stride,
        size_t cn_stride,
        size_t a_offset,const float* zero,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(mr != 0);
    assert(mr <= 4);
    assert(nc != 0);
    assert(kc != 0);
    assert(kc % sizeof(float) == 0);
    assert(ks != 0);
    assert(ks % (4 * sizeof(void*)) == 0);
    assert(a_offset % sizeof(float) == 0);
    assert(a != NULL);
    assert(w != NULL);
    assert(c != NULL);

    float* c0 = c;
    float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
    if XNN_UNPREDICTABLE(mr < 2) {
        c1 = c0;
    }
    float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
    if XNN_UNPREDICTABLE(mr <= 2) {
        c2 = c1;
    }
    float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
    if XNN_UNPREDICTABLE(mr != 4) {
        c3 = c2;
    }

    const float vmin = params->scalar.min;
    const float vmax = params->scalar.max;

    do {
        size_t vl = vsetvl_e32m4(nc); // vector length
        vfloat32m4_t vacc0 = vle32_v_f32m4(w, 16); // 1st row count
        vfloat32m4_t vacc1 = vle32_v_f32m4(w, 16); // 1st row count
        vfloat32m4_t vacc2 = vle32_v_f32m4(w, 16); // 1st row count
        vfloat32m4_t vacc3 = vle32_v_f32m4(w, 16); // 1st row count
        w += 16;

        size_t p = ks;
        size_t kcl = kc / sizeof(float);
        do {
            const float* restrict a0 = a[0];
            assert(a0 != NULL);
            if XNN_UNPREDICTABLE(a0 != zero) {
                a0 = (const float*) ((uintptr_t) a0 + a_offset);
            }
            const float* restrict a1 = a[1];
            assert(a1 != NULL);
            if XNN_UNPREDICTABLE(a1 != zero) {
                a1 = (const float*) ((uintptr_t) a1 + a_offset);
            }
            const float* restrict a2 = a[2];
            assert(a2 != NULL);
            if XNN_UNPREDICTABLE(a2 != zero) {
                a2 = (const float*) ((uintptr_t) a2 + a_offset);
            }
            const float* restrict a3 = a[3];
            assert(a3 != NULL);
            if XNN_UNPREDICTABLE(a3 != zero) {
                a3 = (const float*) ((uintptr_t) a3 + a_offset);
            }
            a += 4;

            size_t k = kc;
            for(size_t k = 0; k < kcl ; k++){
                vfloat32m4_t vw = vle32_v_f32m4(w, 16);
                w += 16;
                vacc0 = vfmacc_vf_f32m4(vacc0, *a0, vw, 16); // update 1st row count
                vacc1 = vfmacc_vf_f32m4(vacc1, *a1, vw, 16); // update 1st row count
                vacc2 = vfmacc_vf_f32m4(vacc2, *a2, vw, 16); // update 1st row count
                vacc3 = vfmacc_vf_f32m4(vacc3, *a3, vw, 16); // update 1st row count
                a0++;
                a1++;
                a2++;
                a3++;
            }
            p -= 4 * sizeof(void*);
        } while (p != 0);
        vacc0 = vfmax_vf_f32m4(vacc0, vmin, vl);
        vacc1 = vfmax_vf_f32m4(vacc1, vmin, vl);
        vacc2 = vfmax_vf_f32m4(vacc2, vmin, vl);
        vacc3 = vfmax_vf_f32m4(vacc3, vmin, vl);

        vacc0 = vfmin_vf_f32m4(vacc0, vmax, vl);
        vacc1 = vfmin_vf_f32m4(vacc1, vmax, vl);
        vacc2 = vfmin_vf_f32m4(vacc2, vmax, vl);
        vacc3 = vfmin_vf_f32m4(vacc3, vmax, vl);

        vse32_v_f32m4(c0, vacc0, vl); // store 1st row result
        vse32_v_f32m4(c1, vacc1, vl); // store 1st row result
        vse32_v_f32m4(c2, vacc2, vl); // store 1st row result
        vse32_v_f32m4(c3, vacc3, vl); // store 1st row result

        if XNN_LIKELY(nc >= 4) {
            c3 = (float*) ((uintptr_t) c3 + cn_stride);
            c2 = (float*) ((uintptr_t) c2 + cn_stride);
            c1 = (float*) ((uintptr_t) c1 + cn_stride);
            c0 = (float*) ((uintptr_t) c0 + cn_stride);

            a = (const float**restrict) ((uintptr_t) a - ks);
        }
        nc -= vl;
    } while (nc != 0);
}

void xnn_f32_igemm_ukernel_1x32__rvv_u8v(
        size_t mr,
        size_t nc,
        size_t kc,
        size_t ks,
        const float** restrict a,
        const float* restrict w,
        float* restrict c,
        size_t cm_stride,
        size_t cn_stride,
        size_t a_offset,
        const float* zero,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
     assert(mr != 0);
    assert(mr <= 1);
    assert(nc != 0);
    assert(kc != 0);
    assert(kc % sizeof(float) == 0);
    assert(ks != 0);
    assert(ks % (1 * sizeof(void*)) == 0);
    assert(a_offset % sizeof(float) == 0);
    assert(a != NULL);
    assert(w != NULL);
    assert(c != NULL);

    float* c0 = c;

    const float vmin = params->scalar.min;
    const float vmax = params->scalar.max;

    do {
        size_t vl = vsetvl_e32m8(nc); // vector length
        vfloat32m8_t vacc0 = vle32_v_f32m8(w, 32); // 1st row count
        w += 32;

        size_t p = ks;
        size_t kcl = kc / sizeof(float);

        do {
             const float* restrict a0 = a[0];
            assert(a0 != NULL);
            if XNN_UNPREDICTABLE(a0 != zero) {
                a0 = (const float*) ((uintptr_t) a0 + a_offset);
            }
            a += 1;

            size_t k = kc;
            for(size_t k = 0; k < kcl ; k++){
                vfloat32m8_t vw = vle32_v_f32m8(w, 32);
                w += 32;
                vacc0 = vfmacc_vf_f32m8(vacc0, *a0, vw, 32); // update 1st row count
                a0++;
            }
            p -= 1 * sizeof(void*);
        } while (p != 0);
        vacc0 = vfmax_vf_f32m8(vacc0, vmin, vl);
        vacc0 = vfmin_vf_f32m8(vacc0, vmax, vl);
        vse32_v_f32m8(c0, vacc0, vl); // store 1st row result

        if XNN_LIKELY(nc >= 4) {
            c0 = (float*) ((uintptr_t) c0 + cn_stride);
            a = (const float**restrict) ((uintptr_t) a - ks);
        }
        nc -= vl;
    } while (nc != 0);
}

void xnn_f32_igemm_minmax_ukernel_4x32__rvv_u8v(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(mr != 0);
    assert(mr <= 4);
    assert(nc != 0);
    assert(kc != 0);
    assert(kc % sizeof(float) == 0);
    assert(ks != 0);
    assert(ks % (4 * sizeof(void*)) == 0);
    assert(a_offset % sizeof(float) == 0);
    assert(a != NULL);
    assert(w != NULL);
    assert(c != NULL);

    float* c0 = c;
    float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
    if XNN_UNPREDICTABLE(mr < 2) {
        c1 = c0;
    }
    float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
    if XNN_UNPREDICTABLE(mr <= 2) {
        c2 = c1;
    }
    float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
    if XNN_UNPREDICTABLE(mr != 4) {
        c3 = c2;
    }

    const float vmin = params->scalar.min;
    const float vmax = params->scalar.max;

    do {
        size_t vl = vsetvl_e32m8(nc); // vector length
        vfloat32m8_t vacc0 = vle32_v_f32m8(w, 32); // 1st row count
        vfloat32m8_t vacc1 = vle32_v_f32m8(w, 32); // 1st row count
        vfloat32m8_t vacc2 = vle32_v_f32m8(w, 32); // 1st row count
        vfloat32m8_t vacc3 = vle32_v_f32m8(w, 32); // 1st row count

        w += 32;

        size_t p = ks;
        size_t kcl = kc / sizeof(float);
        do {
            const float* restrict a0 = a[0];
            assert(a0 != NULL);
            if XNN_UNPREDICTABLE(a0 != zero) {
                a0 = (const float*) ((uintptr_t) a0 + a_offset);
            }
            const float* restrict a1 = a[1];
            assert(a1 != NULL);
            if XNN_UNPREDICTABLE(a1 != zero) {
                a1 = (const float*) ((uintptr_t) a1 + a_offset);
            }
            const float* restrict a2 = a[2];
            assert(a2 != NULL);
            if XNN_UNPREDICTABLE(a2 != zero) {
                a2 = (const float*) ((uintptr_t) a2 + a_offset);
            }
            const float* restrict a3 = a[3];
            assert(a3 != NULL);
            if XNN_UNPREDICTABLE(a3 != zero) {
                a3 = (const float*) ((uintptr_t) a3 + a_offset);
            }
            a += 4;

            size_t k = kc;
            for(size_t k = 0; k < kcl ; k++){
                vfloat32m8_t vw = vle32_v_f32m8(w, 32);
                w += 32;
                vacc0 = vfmacc_vf_f32m8(vacc0, *a0, vw, 32); // update 1st row count
                vacc1 = vfmacc_vf_f32m8(vacc1, *a1, vw, 32); // update 1st row count
                vacc2 = vfmacc_vf_f32m8(vacc2, *a2, vw, 32); // update 1st row count
                vacc3 = vfmacc_vf_f32m8(vacc3, *a3, vw, 32); // update 1st row count
                a0++;
                a1++;
                a2++;
                a3++;
            }
            p -= 4 * sizeof(void*);
        } while (p != 0);
        vacc0 = vfmax_vf_f32m8(vacc0, vmin, vl);
        vacc1 = vfmax_vf_f32m8(vacc1, vmin, vl);
        vacc2 = vfmax_vf_f32m8(vacc2, vmin, vl);
        vacc3 = vfmax_vf_f32m8(vacc3, vmin, vl);

        vacc0 = vfmin_vf_f32m8(vacc0, vmax, vl);
        vacc1 = vfmin_vf_f32m8(vacc1, vmax, vl);
        vacc2 = vfmin_vf_f32m8(vacc2, vmax, vl);
        vacc3 = vfmin_vf_f32m8(vacc3, vmax, vl);
        vse32_v_f32m8(c0, vacc0, vl); // store 1st row result
        vse32_v_f32m8(c1, vacc1, vl); // store 1st row result
        vse32_v_f32m8(c2, vacc2, vl); // store 1st row result
        vse32_v_f32m8(c3, vacc3, vl); // store 1st row result

        if XNN_LIKELY(nc >= 4) {
            c3 = (float*) ((uintptr_t) c3 + cn_stride);
            c2 = (float*) ((uintptr_t) c2 + cn_stride);
            c1 = (float*) ((uintptr_t) c1 + cn_stride);
            c0 = (float*) ((uintptr_t) c0 + cn_stride);

            a = (const float**restrict) ((uintptr_t) a - ks);
        }
        nc -= vl;
    } while (nc != 0);
}



void xnn_f32_igemm_relu_ukernel_1x4__rvv_u1v(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(mr != 0);
    assert(mr <= 1);
    assert(nc != 0);
    assert(kc != 0);
    assert(kc % sizeof(float) == 0);
    assert(ks != 0);
    assert(ks % (1 * sizeof(void*)) == 0);
    assert(a_offset % sizeof(float) == 0);
    assert(a != NULL);
    assert(w != NULL);
    assert(c != NULL);

    float* c0 = c;

    do {
        size_t vl = vsetvl_e32m1(nc); // vector length
        vfloat32m1_t vacc0 = vle32_v_f32m1(w, 4); // 1st row count
        w += 4;

         size_t p = ks;
        size_t kcl = kc / sizeof(float);
        do {
             const float* restrict a0 = a[0];
            assert(a0 != NULL);
            if XNN_UNPREDICTABLE(a0 != zero) {
                a0 = (const float*) ((uintptr_t) a0 + a_offset);
            }
            a += 1;

            size_t k = kc;
            for(size_t k = 0; k < kcl ; k++){
                vfloat32m1_t vw = vle32_v_f32m1(w, 4);
                w += 4;
                vacc0 = vfmacc_vf_f32m1(vacc0, *a0, vw, 4); // update 1st row count
                a0++;
            }
            p -= 1 * sizeof(void*);
        } while (p != 0);
        vacc0 = vfmax_vf_f32m1(vacc0, 0.0f, vl);
        vse32_v_f32m1(c0, vacc0, vl); // store 1st row result

        if XNN_LIKELY(nc >= 4) {
            c0 = (float*) ((uintptr_t) c0 + cn_stride);
            a = (const float**restrict) ((uintptr_t) a - ks);
        }
        nc -= vl;
    } while (nc != 0);
}


void xnn_f32_igemm_relu_ukernel_4x4__rvv_u1v(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,const float* zero,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(mr != 0);
    assert(mr <= 4);
    assert(nc != 0);
    assert(kc != 0);
    assert(kc % sizeof(float) == 0);
    assert(ks != 0);
    assert(ks % (4 * sizeof(void*)) == 0);
    assert(a_offset % sizeof(float) == 0);
    assert(a != NULL);
    assert(w != NULL);
    assert(c != NULL);

    float* c0 = c;
    float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
    if XNN_UNPREDICTABLE(mr < 2) {
        c1 = c0;
    }
    float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
    if XNN_UNPREDICTABLE(mr <= 2) {
        c2 = c1;
    }
    float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
    if XNN_UNPREDICTABLE(mr != 4) {
        c3 = c2;
    }

    do {
        size_t vl = vsetvl_e32m1(nc); // vector length
        vfloat32m1_t vacc0 = vle32_v_f32m1(w, 4); // 1st row count
        vfloat32m1_t vacc1 = vle32_v_f32m1(w, 4); // 1st row count
        vfloat32m1_t vacc2 = vle32_v_f32m1(w, 4); // 1st row count
        vfloat32m1_t vacc3 = vle32_v_f32m1(w, 4); // 1st row count
        w += 4;

        size_t p = ks;
        size_t kcl = kc / sizeof(float);

        do {
            const float* restrict a0 = a[0];
            assert(a0 != NULL);
            if XNN_UNPREDICTABLE(a0 != zero) {
                a0 = (const float*) ((uintptr_t) a0 + a_offset);
            }
            const float* restrict a1 = a[1];
            assert(a1 != NULL);
            if XNN_UNPREDICTABLE(a1 != zero) {
                a1 = (const float*) ((uintptr_t) a1 + a_offset);
            }
            const float* restrict a2 = a[2];
            assert(a2 != NULL);
            if XNN_UNPREDICTABLE(a2 != zero) {
                a2 = (const float*) ((uintptr_t) a2 + a_offset);
            }
            const float* restrict a3 = a[3];
            assert(a3 != NULL);
            if XNN_UNPREDICTABLE(a3 != zero) {
                a3 = (const float*) ((uintptr_t) a3 + a_offset);
            }
            a += 4;

            size_t k = kc;
            for(size_t k = 0; k < kcl ; k++){
                vfloat32m1_t vw = vle32_v_f32m1(w, 4);
                w += 4;
                vacc0 = vfmacc_vf_f32m1(vacc0, *a0, vw, 4); // update 1st row count
                vacc1 = vfmacc_vf_f32m1(vacc1, *a1, vw, 4); // update 1st row count
                vacc2 = vfmacc_vf_f32m1(vacc2, *a2, vw, 4); // update 1st row count
                vacc3 = vfmacc_vf_f32m1(vacc3, *a3, vw, 4); // update 1st row count
                a0++;
                a1++;
                a2++;
                a3++;
            }
            p -= 4 * sizeof(void*);     
        } while (p != 0);
        vacc0 = vfmax_vf_f32m1(vacc0, 0.0f, vl);
        vacc1 = vfmax_vf_f32m1(vacc1, 0.0f, vl);
        vacc2 = vfmax_vf_f32m1(vacc2, 0.0f, vl);
        vacc3 = vfmax_vf_f32m1(vacc3, 0.0f, vl);

        vse32_v_f32m1(c0, vacc0, vl); // store 1st row result
        vse32_v_f32m1(c1, vacc1, vl); // store 1st row result
        vse32_v_f32m1(c2, vacc2, vl); // store 1st row result
        vse32_v_f32m1(c3, vacc3, vl); // store 1st row result

        if XNN_LIKELY(nc >= 4) {
            c3 = (float*) ((uintptr_t) c3 + cn_stride);
            c2 = (float*) ((uintptr_t) c2 + cn_stride);
            c1 = (float*) ((uintptr_t) c1 + cn_stride);
            c0 = (float*) ((uintptr_t) c0 + cn_stride);

            a = (const float**restrict) ((uintptr_t) a - ks);
        }
        nc -= vl;
    } while (nc!=0);
}

void xnn_f32_igemm_relu_ukernel_1x8__rvv_u2v(
        size_t mr,
        size_t nc,
        size_t kc,
        size_t ks,
        const float** restrict a,
        const float* restrict w,
        float* restrict c,
        size_t cm_stride,
        size_t cn_stride,
        size_t a_offset,
        const float* zero,
        const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(mr != 0);
    assert(mr <= 1);
    assert(nc != 0);
    assert(kc != 0);
    assert(kc % sizeof(float) == 0);
    assert(ks != 0);
    assert(ks % (1 * sizeof(void*)) == 0);
    assert(a_offset % sizeof(float) == 0);
    assert(a != NULL);
    assert(w != NULL);
    assert(c != NULL);

    float* c0 = c;

    do {
        size_t vl = vsetvl_e32m2(nc); // vector length
        vfloat32m2_t vacc0 = vle32_v_f32m2(w, 8); // 1st row count
        w += 8;

        size_t p = ks;
        size_t kcl = kc / sizeof(float);
        do {
            const float* restrict a0 = a[0];
            assert(a0 != NULL);
            if XNN_UNPREDICTABLE(a0 != zero) {
                a0 = (const float*) ((uintptr_t) a0 + a_offset);
            }
            a += 1;

            size_t k = kc;
            for(size_t k = 0; k < kcl ; k++){
                vfloat32m2_t vw = vle32_v_f32m2(w, 8);
                w += 8;
                vacc0 = vfmacc_vf_f32m2(vacc0, *a0, vw, 8); // update 1st row count
                a0++;
            }
            p -= 1 * sizeof(void*);
        } while (p != 0);
        vacc0 = vfmax_vf_f32m2(vacc0, 0.0f, vl);
        vse32_v_f32m2(c0, vacc0, vl); // store 1st row result

        if XNN_LIKELY(nc >= 4) {
            c0 = (float*) ((uintptr_t) c0 + cn_stride);
            a = (const float**restrict) ((uintptr_t) a - ks);
        }
        nc -= vl;
    } while (nc != 0);
}


void xnn_f32_igemm_relu_ukernel_4x8__rvv_u2v(
        size_t mr,
        size_t nc,
        size_t kc,
        size_t ks,
        const float** restrict a,
        const float* restrict w,
        float* restrict c,
        size_t cm_stride,
        size_t cn_stride,
        size_t a_offset,const float* zero,
        const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(mr != 0);
    assert(mr <= 4);
    assert(nc != 0);
    assert(kc != 0);
    assert(kc % sizeof(float) == 0);
    assert(ks != 0);
    assert(ks % (4 * sizeof(void*)) == 0);
    assert(a_offset % sizeof(float) == 0);
    assert(a != NULL);
    assert(w != NULL);
    assert(c != NULL);

    float* c0 = c;
    float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
    if XNN_UNPREDICTABLE(mr < 2) {
        c1 = c0;
    }
    float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
    if XNN_UNPREDICTABLE(mr <= 2) {
        c2 = c1;
    }
    float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
    if XNN_UNPREDICTABLE(mr != 4) {
        c3 = c2;
    }

    do {
        size_t vl = vsetvl_e32m2(nc); // vector length
        vfloat32m2_t vacc0 = vle32_v_f32m2(w, 8); // 1st row count
        vfloat32m2_t vacc1 = vle32_v_f32m2(w, 8); // 1st row count
        vfloat32m2_t vacc2 = vle32_v_f32m2(w, 8); // 1st row count
        vfloat32m2_t vacc3 = vle32_v_f32m2(w, 8); // 1st row count
        w += 8;

        size_t p = ks;
        size_t kcl = kc / sizeof(float);
        do {
            const float* restrict a0 = a[0];
            assert(a0 != NULL);
            if XNN_UNPREDICTABLE(a0 != zero) {
                a0 = (const float*) ((uintptr_t) a0 + a_offset);
            }
            const float* restrict a1 = a[1];
            assert(a1 != NULL);
            if XNN_UNPREDICTABLE(a1 != zero) {
                a1 = (const float*) ((uintptr_t) a1 + a_offset);
            }
            const float* restrict a2 = a[2];
            assert(a2 != NULL);
            if XNN_UNPREDICTABLE(a2 != zero) {
                a2 = (const float*) ((uintptr_t) a2 + a_offset);
            }
            const float* restrict a3 = a[3];
            assert(a3 != NULL);
            if XNN_UNPREDICTABLE(a3 != zero) {
                a3 = (const float*) ((uintptr_t) a3 + a_offset);
            }
            a += 4;

            size_t k = kc;
            for(size_t k = 0; k < kcl ; k++){
                vfloat32m2_t vw = vle32_v_f32m2(w, 8);
                w += 8;
                vacc0 = vfmacc_vf_f32m2(vacc0, *a0, vw, 8); // update 1st row count
                vacc1 = vfmacc_vf_f32m2(vacc1, *a1, vw, 8); // update 1st row count
                vacc2 = vfmacc_vf_f32m2(vacc2, *a2, vw, 8); // update 1st row count
                vacc3 = vfmacc_vf_f32m2(vacc3, *a3, vw, 8); // update 1st row count
                a0++;
                a1++;
                a2++;
                a3++;
            }
            p -= 4 * sizeof(void*);
        } while (p != 0);
        vacc0 = vfmax_vf_f32m2(vacc0, 0.0f, vl);
        vacc1 = vfmax_vf_f32m2(vacc1, 0.0f, vl);
        vacc2 = vfmax_vf_f32m2(vacc2, 0.0f, vl);
        vacc3 = vfmax_vf_f32m2(vacc3, 0.0f, vl);

        vse32_v_f32m2(c0, vacc0, vl); // store 1st row result
        vse32_v_f32m2(c1, vacc1, vl); // store 1st row result
        vse32_v_f32m2(c2, vacc2, vl); // store 1st row result
        vse32_v_f32m2(c3, vacc3, vl); // store 1st row result

        if XNN_LIKELY(nc >= 4) {
            c3 = (float*) ((uintptr_t) c3 + cn_stride);
            c2 = (float*) ((uintptr_t) c2 + cn_stride);
            c1 = (float*) ((uintptr_t) c1 + cn_stride);
            c0 = (float*) ((uintptr_t) c0 + cn_stride);

            a = (const float**restrict) ((uintptr_t) a - ks);
        }
        nc -= vl;
    } while (nc != 0);
}

void xnn_f32_igemm_relu_ukernel_1x16__rvv_u4v(
        size_t mr,
        size_t nc,
        size_t kc,
        size_t ks,
        const float** restrict a,
        const float* restrict w,
        float* restrict c,
        size_t cm_stride,
        size_t cn_stride,
        size_t a_offset,
        const float* zero,
        const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(mr != 0);
    assert(mr <= 1);
    assert(nc != 0);
    assert(kc != 0);
    assert(kc % sizeof(float) == 0);
    assert(ks != 0);
    assert(ks % (1 * sizeof(void*)) == 0);
    assert(a_offset % sizeof(float) == 0);
    assert(a!= NULL);
    assert(w!= NULL);
    assert(c!= NULL);

    float* c0 = c;

    do {
        size_t vl = vsetvl_e32m4(nc); // vector length
        vfloat32m4_t vacc0 = vle32_v_f32m4(w, 16); // 1st row count
        w += 16;

        size_t p = ks;
        size_t kcl = kc / sizeof(float);
        do {
            const float* restrict a0 = a[0];
            assert(a0 != NULL);
            if XNN_UNPREDICTABLE(a0 != zero) {
                a0 = (const float*) ((uintptr_t) a0 + a_offset);
            }
            a += 1;

            size_t k = kc;
            for(size_t k = 0; k < kcl ; k++){
                vfloat32m4_t vw = vle32_v_f32m4(w, 16);
                w += 16;
                vacc0 = vfmacc_vf_f32m4(vacc0, *a0, vw, 16); // update 1st row count
                a0++;
            }
            p -= 1 * sizeof(void*);
        } while (p != 0);
        vacc0 = vfmax_vf_f32m4(vacc0, 0.0f, vl);
        vse32_v_f32m4(c0, vacc0, vl); // store 1st row result

        if XNN_LIKELY(nc >= 4) {
            c0 = (float*) ((uintptr_t) c0 + cn_stride);
            a = (const float**restrict) ((uintptr_t) a - ks);
        }
        nc -= vl;
    } while (nc != 0);
}

void xnn_f32_igemm_relu_ukernel_4x16__rvv_u4v(
        size_t mr,
        size_t nc,
        size_t kc,
        size_t ks,
        const float** restrict a,
        const float* restrict w,
        float* restrict c,
        size_t cm_stride,
        size_t cn_stride,
        size_t a_offset,const float* zero,
        const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(mr != 0);
    assert(mr <= 4);
    assert(nc != 0);
    assert(kc != 0);
    assert(kc % sizeof(float) == 0);
    assert(ks != 0);
    assert(ks % (4 * sizeof(void*)) == 0);
    assert(a_offset % sizeof(float) == 0);
    assert(a != NULL);
    assert(w != NULL);
    assert(c != NULL);

    float* c0 = c;
    float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
    if XNN_UNPREDICTABLE(mr < 2) {
        c1 = c0;
    }
    float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
    if XNN_UNPREDICTABLE(mr <= 2) {
        c2 = c1;
    }
    float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
    if XNN_UNPREDICTABLE(mr != 4) {
        c3 = c2;
    }

    do {
        size_t vl = vsetvl_e32m4(nc); // vector length
        vfloat32m4_t vacc0 = vle32_v_f32m4(w, 16); // 1st row count
        vfloat32m4_t vacc1 = vle32_v_f32m4(w, 16); // 1st row count
        vfloat32m4_t vacc2 = vle32_v_f32m4(w, 16); // 1st row count
        vfloat32m4_t vacc3 = vle32_v_f32m4(w, 16); // 1st row count
        w += 16;

        size_t p = ks;
        size_t kcl = kc / sizeof(float);
        do {
            const float* restrict a0 = a[0];
            assert(a0 != NULL);
            if XNN_UNPREDICTABLE(a0 != zero) {
                a0 = (const float*) ((uintptr_t) a0 + a_offset);
            }
            const float* restrict a1 = a[1];
            assert(a1 != NULL);
            if XNN_UNPREDICTABLE(a1 != zero) {
                a1 = (const float*) ((uintptr_t) a1 + a_offset);
            }
            const float* restrict a2 = a[2];
            assert(a2 != NULL);
            if XNN_UNPREDICTABLE(a2 != zero) {
                a2 = (const float*) ((uintptr_t) a2 + a_offset);
            }
            const float* restrict a3 = a[3];
            assert(a3 != NULL);
            if XNN_UNPREDICTABLE(a3 != zero) {
                a3 = (const float*) ((uintptr_t) a3 + a_offset);
            }
            a += 4;

            size_t k = kc;
            for(size_t k = 0; k < kcl ; k++){
                vfloat32m4_t vw = vle32_v_f32m4(w, 16);
                w += 16;
                vacc0 = vfmacc_vf_f32m4(vacc0, *a0, vw, 16); // update 1st row count
                vacc1 = vfmacc_vf_f32m4(vacc1, *a1, vw, 16); // update 1st row count
                vacc2 = vfmacc_vf_f32m4(vacc2, *a2, vw, 16); // update 1st row count
                vacc3 = vfmacc_vf_f32m4(vacc3, *a3, vw, 16); // update 1st row count
                a0++;
                a1++;
                a2++;
                a3++;
            }
            p -= 4 * sizeof(void*);
        } while (p != 0);
        vacc0 = vfmax_vf_f32m4(vacc0, 0.0f, vl);
        vacc1 = vfmax_vf_f32m4(vacc1, 0.0f, vl);
        vacc2 = vfmax_vf_f32m4(vacc2, 0.0f, vl);
        vacc3 = vfmax_vf_f32m4(vacc3, 0.0f, vl);

        vse32_v_f32m4(c0, vacc0, vl); // store 1st row result
        vse32_v_f32m4(c1, vacc1, vl); // store 1st row result
        vse32_v_f32m4(c2, vacc2, vl); // store 1st row result
        vse32_v_f32m4(c3, vacc3, vl); // store 1st row result

        if XNN_LIKELY(nc >= 4) {
            c3 = (float*) ((uintptr_t) c3 + cn_stride);
            c2 = (float*) ((uintptr_t) c2 + cn_stride);
            c1 = (float*) ((uintptr_t) c1 + cn_stride);
            c0 = (float*) ((uintptr_t) c0 + cn_stride);

            a = (const float**restrict) ((uintptr_t) a - ks);
        }
        nc -= vl;
    } while (nc != 0);
}


void xnn_f32_igemm_relu_ukernel_1x32__rvv_u8v(
        size_t mr,
        size_t nc,
        size_t kc,
        size_t ks,
        const float** restrict a,
        const float* restrict w,
        float* restrict c,
        size_t cm_stride,
        size_t cn_stride,
        size_t a_offset,
        const float* zero,
        const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
     assert(mr != 0);
    assert(mr <= 1);
    assert(nc != 0);
    assert(kc != 0);
    assert(kc % sizeof(float) == 0);
    assert(ks != 0);
    assert(ks % (1 * sizeof(void*)) == 0);
    assert(a_offset % sizeof(float) == 0);
    assert(a != NULL);
    assert(w != NULL);
    assert(c != NULL);

    float* c0 = c;

    do {
        size_t vl = vsetvl_e32m8(nc); // vector length
        vfloat32m8_t vacc0 = vle32_v_f32m8(w, 32); // 1st row count
        w += 32;

        size_t p = ks;
        size_t kcl = kc / sizeof(float);

        do {
             const float* restrict a0 = a[0];
            assert(a0 != NULL);
            if XNN_UNPREDICTABLE(a0 != zero) {
                a0 = (const float*) ((uintptr_t) a0 + a_offset);
            }
            a += 1;

            size_t k = kc;
            for(size_t k = 0; k < kcl ; k++){
                vfloat32m8_t vw = vle32_v_f32m8(w, 32);
                w += 32;
                vacc0 = vfmacc_vf_f32m8(vacc0, *a0, vw, 32); // update 1st row count
                a0++;
            }
            p -= 1 * sizeof(void*);
        } while (p != 0);
        vacc0 = vfmax_vf_f32m8(vacc0, 0.0f, vl);
        vse32_v_f32m8(c0, vacc0, vl); // store 1st row result

        if XNN_LIKELY(nc >= 4) {
            c0 = (float*) ((uintptr_t) c0 + cn_stride);
            a = (const float**restrict) ((uintptr_t) a - ks);
        }
        nc -= vl;
    } while (nc != 0);
}


void xnn_f32_igemm_relu_ukernel_4x32__rvv_u8v(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,const float* zero,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(mr != 0);
    assert(mr <= 4);
    assert(nc != 0);
    assert(kc != 0);
    assert(kc % sizeof(float) == 0);
    assert(ks != 0);
    assert(ks % (4 * sizeof(void*)) == 0);
    assert(a_offset % sizeof(float) == 0);
    assert(a != NULL);
    assert(w != NULL);
    assert(c != NULL);

    float* c0 = c;
    float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
    if XNN_UNPREDICTABLE(mr < 2) {
        c1 = c0;
    }
    float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
    if XNN_UNPREDICTABLE(mr <= 2) {
        c2 = c1;
    }
    float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
    if XNN_UNPREDICTABLE(mr != 4) {
        c3 = c2;
    }

    do {
        size_t vl = vsetvl_e32m8(nc); // vector length
        vfloat32m8_t vacc0 = vle32_v_f32m8(w, 32); // 1st row count
        vfloat32m8_t vacc1 = vle32_v_f32m8(w, 32); // 1st row count
        vfloat32m8_t vacc2 = vle32_v_f32m8(w, 32); // 1st row count
        vfloat32m8_t vacc3 = vle32_v_f32m8(w, 32); // 1st row count

        w += 32;

        size_t p = ks;
        size_t kcl = kc / sizeof(float);
        do {
            const float* restrict a0 = a[0];
            assert(a0 != NULL);
            if XNN_UNPREDICTABLE(a0 != zero) {
                a0 = (const float*) ((uintptr_t) a0 + a_offset);
            }
            const float* restrict a1 = a[1];
            assert(a1 != NULL);
            if XNN_UNPREDICTABLE(a1 != zero) {
                a1 = (const float*) ((uintptr_t) a1 + a_offset);
            }
            const float* restrict a2 = a[2];
            assert(a2 != NULL);
            if XNN_UNPREDICTABLE(a2 != zero) {
                a2 = (const float*) ((uintptr_t) a2 + a_offset);
            }
            const float* restrict a3 = a[3];
            assert(a3 != NULL);
            if XNN_UNPREDICTABLE(a3 != zero) {
                a3 = (const float*) ((uintptr_t) a3 + a_offset);
            }
            a += 4;

            size_t k = kc;
            for(size_t k = 0; k < kcl ; k++){
                vfloat32m8_t vw = vle32_v_f32m8(w, 32);
                w += 32;
                vacc0 = vfmacc_vf_f32m8(vacc0, *a0, vw, 32); // update 1st row count
                vacc1 = vfmacc_vf_f32m8(vacc1, *a1, vw, 32); // update 1st row count
                vacc2 = vfmacc_vf_f32m8(vacc2, *a2, vw, 32); // update 1st row count
                vacc3 = vfmacc_vf_f32m8(vacc3, *a3, vw, 32); // update 1st row count
                a0++;
                a1++;
                a2++;
                a3++;
            }
            p -= 4 * sizeof(void*);
        } while (p != 0);
        vacc0 = vfmax_vf_f32m8(vacc0, 0.0f, vl);
        vacc1 = vfmax_vf_f32m8(vacc1, 0.0f, vl);
        vacc2 = vfmax_vf_f32m8(vacc2, 0.0f, vl);
        vacc3 = vfmax_vf_f32m8(vacc3, 0.0f, vl);
        vse32_v_f32m8(c0, vacc0, vl); // store 1st row result
        vse32_v_f32m8(c1, vacc1, vl); // store 1st row result
        vse32_v_f32m8(c2, vacc2, vl); // store 1st row result
        vse32_v_f32m8(c3, vacc3, vl); // store 1st row result

        if XNN_LIKELY(nc >= 4) {
            c3 = (float*) ((uintptr_t) c3 + cn_stride);
            c2 = (float*) ((uintptr_t) c2 + cn_stride);
            c1 = (float*) ((uintptr_t) c1 + cn_stride);
            c0 = (float*) ((uintptr_t) c0 + cn_stride);

            a = (const float**restrict) ((uintptr_t) a - ks);
        }
        nc -= vl;
    } while (nc != 0);
}



//u2v

void xnn_f32_maxpool_minmax_ukernel_9p8x__rvv_u2v(
        size_t output_pixels,
        size_t kernel_elements,
        size_t channels,
        const float** input,
        size_t input_offset,
        float* output,
        size_t input_increment,
        size_t output_increment,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(output_pixels != 0);
    assert(kernel_elements != 0);
    assert(channels != 0);

    const float voutput_min = params->scalar.min;
    const float voutput_max = params->scalar.max;
    do {
        float* o = output;
        {
            const float* i0 = *input++;
            const float* i1 = *input++;
            const float* i2 = *input++;
            const float* i3 = *input++;
            const float* i4 = *input++;
            const float* i5 = *input++;
            const float* i6 = *input++;
            const float* i7 = *input++;
            const float* i8 = *input++;
            i0 = (const float*) ((uintptr_t) i0 + input_offset);
            i1 = (const float*) ((uintptr_t) i1 + input_offset);
            i2 = (const float*) ((uintptr_t) i2 + input_offset);
            i3 = (const float*) ((uintptr_t) i3 + input_offset);
            i4 = (const float*) ((uintptr_t) i4 + input_offset);
            i5 = (const float*) ((uintptr_t) i5 + input_offset);
            i6 = (const float*) ((uintptr_t) i6 + input_offset);
            i7 = (const float*) ((uintptr_t) i7 + input_offset);
            i8 = (const float*) ((uintptr_t) i8 + input_offset);
            if (kernel_elements < 2) {
                i1 = i0;
            }
            if (kernel_elements <= 2) {
                i2 = i0;
            }
            if (kernel_elements < 4) {
                i3 = i0;
            }
            if (kernel_elements <= 4) {
                i4 = i0;
            }
            if (kernel_elements < 6) {
                i5 = i0;
            }
            if (kernel_elements <= 6) {
                i6 = i0;
            }
            if (kernel_elements < 8) {
                i7 = i0;
            }
            if (kernel_elements <= 8) {
                i8 = i0;
            }

            size_t c = channels;
            do {
                size_t vl = vsetvl_e32m2(c);
                vfloat32m2_t vi0 = vle32_v_f32m2(i0, vl); i0 += vl;
                vfloat32m2_t vi1 = vle32_v_f32m2(i1, vl); i1 += vl;
                vfloat32m2_t vi2 = vle32_v_f32m2(i2, vl); i2 += vl;
                vfloat32m2_t vi3 = vle32_v_f32m2(i3, vl); i3 += vl;
                vfloat32m2_t vi4 = vle32_v_f32m2(i4, vl); i4 += vl;
                vfloat32m2_t vi5 = vle32_v_f32m2(i5, vl); i5 += vl;
                vfloat32m2_t vi6 = vle32_v_f32m2(i6, vl); i6 += vl;
                vfloat32m2_t vi7 = vle32_v_f32m2(i7, vl); i7 += vl;
                vfloat32m2_t vi8 = vle32_v_f32m2(i8, vl); i8 += vl;

                vfloat32m2_t vmax01 = vfmax_vv_f32m2(vi0, vi1, vl);
                vfloat32m2_t vmax23 = vfmax_vv_f32m2(vi2, vi3, vl);
                vfloat32m2_t vmax45 = vfmax_vv_f32m2(vi4, vi5, vl);
                vfloat32m2_t vmax67 = vfmax_vv_f32m2(vi6, vi7, vl);
                vfloat32m2_t vmax018 = vfmax_vv_f32m2(vmax01, vi8, vl);

                vfloat32m2_t vmax2345 = vfmax_vv_f32m2(vmax23, vmax45, vl);
                vfloat32m2_t vmax01678 = vfmax_vv_f32m2(vmax018, vmax67, vl);
                vfloat32m2_t vout = vfmax_vv_f32m2(vmax2345, vmax01678, vl);
                vout = vfmax_vf_f32m2(vout, voutput_min, vl);
                vout = vfmin_vf_f32m2(vout, voutput_max, vl);

                vse32_v_f32m2(o, vout, vl);

                o += vl;
                c -= vl;
            } while (c != 0);
        }

        for (ptrdiff_t k = (ptrdiff_t) kernel_elements - 9; k > 0; k -= 8) {
            const float* i0 = *input++;
            const float* i1 = *input++;
            const float* i2 = *input++;
            const float* i3 = *input++;
            const float* i4 = *input++;
            const float* i5 = *input++;
            const float* i6 = *input++;
            const float* i7 = *input++;
            i0 = (const float*) ((uintptr_t) i0 + input_offset);
            i1 = (const float*) ((uintptr_t) i1 + input_offset);
            i2 = (const float*) ((uintptr_t) i2 + input_offset);
            i3 = (const float*) ((uintptr_t) i3 + input_offset);
            i4 = (const float*) ((uintptr_t) i4 + input_offset);
            i5 = (const float*) ((uintptr_t) i5 + input_offset);
            i6 = (const float*) ((uintptr_t) i6 + input_offset);
            i7 = (const float*) ((uintptr_t) i7 + input_offset);
            if (k < 2) {
                i1 = i0;
            }
            if (k <= 2) {
                i2 = i0;
            }
            if (k < 4) {
                i3 = i0;
            }
            if (k <= 4) {
                i4 = i0;
            }
            if (k < 6) {
                i5 = i0;
            }
            if (k <= 6) {
                i6 = i0;
            }
            if (k < 8) {
                i7 = i0;
            }

            o = output;
            size_t c = channels;
            do {
                size_t vl = vsetvl_e32m2(c);
                vfloat32m2_t vi0 = vle32_v_f32m2(i0, vl); i0 += vl;
                vfloat32m2_t vi1 = vle32_v_f32m2(i1, vl); i1 += vl;
                vfloat32m2_t vi2 = vle32_v_f32m2(i2, vl); i2 += vl;
                vfloat32m2_t vi3 = vle32_v_f32m2(i3, vl); i3 += vl;
                vfloat32m2_t vi4 = vle32_v_f32m2(i4, vl); i4 += vl;
                vfloat32m2_t vi5 = vle32_v_f32m2(i5, vl); i5 += vl;
                vfloat32m2_t vi6 = vle32_v_f32m2(i6, vl); i6 += vl;
                vfloat32m2_t vi7 = vle32_v_f32m2(i7, vl); i7 += vl;
                vfloat32m2_t vi8 = vle32_v_f32m2(o, vl);

                vfloat32m2_t vmax01 = vfmax_vv_f32m2(vi0, vi1, vl);
                vfloat32m2_t vmax23 = vfmax_vv_f32m2(vi2, vi3, vl);
                vfloat32m2_t vmax45 = vfmax_vv_f32m2(vi4, vi5, vl);
                vfloat32m2_t vmax67 = vfmax_vv_f32m2(vi6, vi7, vl);
                vfloat32m2_t vmax018 = vfmax_vv_f32m2(vmax01, vi8, vl);

                vfloat32m2_t vmax2345 = vfmax_vv_f32m2(vmax23, vmax45, vl);
                vfloat32m2_t vmax01678 = vfmax_vv_f32m2(vmax018, vmax67, vl);
                vfloat32m2_t vout = vfmax_vv_f32m2(vmax2345, vmax01678, vl);
                vout = vfmax_vf_f32m2(vout, voutput_min, vl);
                vout = vfmin_vf_f32m2(vout, voutput_max, vl);

                vse32_v_f32m2(o, vout, vl);

                o += vl;
                c -= vl;
            } while (c != 0);
        }
        input = (const float**) ((uintptr_t) input + input_increment);
        output = (float*) ((uintptr_t) o + output_increment);
    } while (--output_pixels != 0);
}

void xnn_f32_vhswish_ukernel__rvv_u2v(
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

    size_t size = batch / sizeof(float);
    do {
        // 动态设置向量长度
        const size_t vl = vsetvl_e32m2(size);

        // 加载输入向量
        vfloat32m2_t vx = vle32_v_f32m2(input, vl);
        input += vl;

        vfloat32m2_t vacc = vfadd_vf_f32m2(vx, vthree, vl);
        vx = vfmul_vf_f32m2(vx, vsixth, vl);

        // 应用最小值约束
        vacc = vfmax_vf_f32m2(vacc, vzero, vl);

        // 应用最大值约束
        vacc = vfmin_vf_f32m2(vacc, vsix, vl);

        vacc = vfmul_vv_f32m2(vacc, vx, vl);

        // 存储结果
        vse32_v_f32m2(output, vacc, vl);
        output += vl;
        size -= vl;
    } while (size > 0);
}




//u1v

void xnn_f32_maxpool_minmax_ukernel_9p8x__rvv_u1v(
        size_t output_pixels,
        size_t kernel_elements,
        size_t channels,
        const float** input,
        size_t input_offset,
        float* output,
        size_t input_increment,
        size_t output_increment,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(output_pixels != 0);
    assert(kernel_elements != 0);
    assert(channels != 0);

    const float voutput_min = params->scalar.min;
    const float voutput_max = params->scalar.max;
    do {
        float* o = output;
        {
            const float* i0 = *input++;
            const float* i1 = *input++;
            const float* i2 = *input++;
            const float* i3 = *input++;
            const float* i4 = *input++;
            const float* i5 = *input++;
            const float* i6 = *input++;
            const float* i7 = *input++;
            const float* i8 = *input++;
            i0 = (const float*) ((uintptr_t) i0 + input_offset);
            i1 = (const float*) ((uintptr_t) i1 + input_offset);
            i2 = (const float*) ((uintptr_t) i2 + input_offset);
            i3 = (const float*) ((uintptr_t) i3 + input_offset);
            i4 = (const float*) ((uintptr_t) i4 + input_offset);
            i5 = (const float*) ((uintptr_t) i5 + input_offset);
            i6 = (const float*) ((uintptr_t) i6 + input_offset);
            i7 = (const float*) ((uintptr_t) i7 + input_offset);
            i8 = (const float*) ((uintptr_t) i8 + input_offset);
            if (kernel_elements < 2) {
                i1 = i0;
            }
            if (kernel_elements <= 2) {
                i2 = i0;
            }
            if (kernel_elements < 4) {
                i3 = i0;
            }
            if (kernel_elements <= 4) {
                i4 = i0;
            }
            if (kernel_elements < 6) {
                i5 = i0;
            }
            if (kernel_elements <= 6) {
                i6 = i0;
            }
            if (kernel_elements < 8) {
                i7 = i0;
            }
            if (kernel_elements <= 8) {
                i8 = i0;
            }

            size_t c = channels;
            do {
                size_t vl = vsetvl_e32m1(c);
                vfloat32m1_t vi0 = vle32_v_f32m1(i0, vl); i0 += vl;
                vfloat32m1_t vi1 = vle32_v_f32m1(i1, vl); i1 += vl;
                vfloat32m1_t vi2 = vle32_v_f32m1(i2, vl); i2 += vl;
                vfloat32m1_t vi3 = vle32_v_f32m1(i3, vl); i3 += vl;
                vfloat32m1_t vi4 = vle32_v_f32m1(i4, vl); i4 += vl;
                vfloat32m1_t vi5 = vle32_v_f32m1(i5, vl); i5 += vl;
                vfloat32m1_t vi6 = vle32_v_f32m1(i6, vl); i6 += vl;
                vfloat32m1_t vi7 = vle32_v_f32m1(i7, vl); i7 += vl;
                vfloat32m1_t vi8 = vle32_v_f32m1(i8, vl); i8 += vl;

                vfloat32m1_t vmax01 = vfmax_vv_f32m1(vi0, vi1, vl);
                vfloat32m1_t vmax23 = vfmax_vv_f32m1(vi2, vi3, vl);
                vfloat32m1_t vmax45 = vfmax_vv_f32m1(vi4, vi5, vl);
                vfloat32m1_t vmax67 = vfmax_vv_f32m1(vi6, vi7, vl);
                vfloat32m1_t vmax018 = vfmax_vv_f32m1(vmax01, vi8, vl);

                vfloat32m1_t vmax2345 = vfmax_vv_f32m1(vmax23, vmax45, vl);
                vfloat32m1_t vmax01678 = vfmax_vv_f32m1(vmax018, vmax67, vl);
                vfloat32m1_t vout = vfmax_vv_f32m1(vmax2345, vmax01678, vl);
                vout = vfmax_vf_f32m1(vout, voutput_min, vl);
                vout = vfmin_vf_f32m1(vout, voutput_max, vl);

                vse32_v_f32m1(o, vout, vl);

                o += vl;
                c -= vl;
            } while (c != 0);
        }

        for (ptrdiff_t k = (ptrdiff_t) kernel_elements - 9; k > 0; k -= 8) {
            const float* i0 = *input++;
            const float* i1 = *input++;
            const float* i2 = *input++;
            const float* i3 = *input++;
            const float* i4 = *input++;
            const float* i5 = *input++;
            const float* i6 = *input++;
            const float* i7 = *input++;
            i0 = (const float*) ((uintptr_t) i0 + input_offset);
            i1 = (const float*) ((uintptr_t) i1 + input_offset);
            i2 = (const float*) ((uintptr_t) i2 + input_offset);
            i3 = (const float*) ((uintptr_t) i3 + input_offset);
            i4 = (const float*) ((uintptr_t) i4 + input_offset);
            i5 = (const float*) ((uintptr_t) i5 + input_offset);
            i6 = (const float*) ((uintptr_t) i6 + input_offset);
            i7 = (const float*) ((uintptr_t) i7 + input_offset);
            if (k < 2) {
                i1 = i0;
            }
            if (k <= 2) {
                i2 = i0;
            }
            if (k < 4) {
                i3 = i0;
            }
            if (k <= 4) {
                i4 = i0;
            }
            if (k < 6) {
                i5 = i0;
            }
            if (k <= 6) {
                i6 = i0;
            }
            if (k < 8) {
                i7 = i0;
            }

            o = output;
            size_t c = channels;
            do {
                size_t vl = vsetvl_e32m1(c);
                vfloat32m1_t vi0 = vle32_v_f32m1(i0, vl); i0 += vl;
                vfloat32m1_t vi1 = vle32_v_f32m1(i1, vl); i1 += vl;
                vfloat32m1_t vi2 = vle32_v_f32m1(i2, vl); i2 += vl;
                vfloat32m1_t vi3 = vle32_v_f32m1(i3, vl); i3 += vl;
                vfloat32m1_t vi4 = vle32_v_f32m1(i4, vl); i4 += vl;
                vfloat32m1_t vi5 = vle32_v_f32m1(i5, vl); i5 += vl;
                vfloat32m1_t vi6 = vle32_v_f32m1(i6, vl); i6 += vl;
                vfloat32m1_t vi7 = vle32_v_f32m1(i7, vl); i7 += vl;
                vfloat32m1_t vi8 = vle32_v_f32m1(o, vl);

                vfloat32m1_t vmax01 = vfmax_vv_f32m1(vi0, vi1, vl);
                vfloat32m1_t vmax23 = vfmax_vv_f32m1(vi2, vi3, vl);
                vfloat32m1_t vmax45 = vfmax_vv_f32m1(vi4, vi5, vl);
                vfloat32m1_t vmax67 = vfmax_vv_f32m1(vi6, vi7, vl);
                vfloat32m1_t vmax018 = vfmax_vv_f32m1(vmax01, vi8, vl);

                vfloat32m1_t vmax2345 = vfmax_vv_f32m1(vmax23, vmax45, vl);
                vfloat32m1_t vmax01678 = vfmax_vv_f32m1(vmax018, vmax67, vl);
                vfloat32m1_t vout = vfmax_vv_f32m1(vmax2345, vmax01678, vl);
                vout = vfmax_vf_f32m1(vout, voutput_min, vl);
                vout = vfmin_vf_f32m1(vout, voutput_max, vl);

                vse32_v_f32m1(o, vout, vl);

                o += vl;
                c -= vl;
            } while (c != 0);
        }
        input = (const float**) ((uintptr_t) input + input_increment);
        output = (float*) ((uintptr_t) o + output_increment);
    } while (--output_pixels != 0);
}

void xnn_f32_vhswish_ukernel__rvv_u1v(
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

    size_t size = batch / sizeof(float);
    do {
        // 动态设置向量长度
        const size_t vl = vsetvl_e32m1(size);

        // 加载输入向量
        vfloat32m1_t vx = vle32_v_f32m1(input, vl);
        input += vl;

        vfloat32m1_t vacc = vfadd_vf_f32m1(vx, vthree, vl);
        vx = vfmul_vf_f32m1(vx, vsixth, vl);

        // 应用最小值约束
        vacc = vfmax_vf_f32m1(vacc, vzero, vl);

        // 应用最大值约束
        vacc = vfmin_vf_f32m1(vacc, vsix, vl);

        vacc = vfmul_vv_f32m1(vacc, vx, vl);

        // 存储结果
        vse32_v_f32m1(output, vacc, vl);
        output += vl;
        size -= vl;
    } while (size > 0);
}


//u4v

void xnn_f32_maxpool_minmax_ukernel_9p8x__rvv_u4v(
        size_t output_pixels,
        size_t kernel_elements,
        size_t channels,
        const float** input,
        size_t input_offset,
        float* output,
        size_t input_increment,
        size_t output_increment,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(output_pixels != 0);
    assert(kernel_elements != 0);
    assert(channels != 0);

    const float voutput_min = params->scalar.min;
    const float voutput_max = params->scalar.max;
    do {
        float* o = output;
        {
            const float* i0 = *input++;
            const float* i1 = *input++;
            const float* i2 = *input++;
            const float* i3 = *input++;
            const float* i4 = *input++;
            const float* i5 = *input++;
            const float* i6 = *input++;
            const float* i7 = *input++;
            const float* i8 = *input++;
            i0 = (const float*) ((uintptr_t) i0 + input_offset);
            i1 = (const float*) ((uintptr_t) i1 + input_offset);
            i2 = (const float*) ((uintptr_t) i2 + input_offset);
            i3 = (const float*) ((uintptr_t) i3 + input_offset);
            i4 = (const float*) ((uintptr_t) i4 + input_offset);
            i5 = (const float*) ((uintptr_t) i5 + input_offset);
            i6 = (const float*) ((uintptr_t) i6 + input_offset);
            i7 = (const float*) ((uintptr_t) i7 + input_offset);
            i8 = (const float*) ((uintptr_t) i8 + input_offset);
            if (kernel_elements < 2) {
                i1 = i0;
            }
            if (kernel_elements <= 2) {
                i2 = i0;
            }
            if (kernel_elements < 4) {
                i3 = i0;
            }
            if (kernel_elements <= 4) {
                i4 = i0;
            }
            if (kernel_elements < 6) {
                i5 = i0;
            }
            if (kernel_elements <= 6) {
                i6 = i0;
            }
            if (kernel_elements < 8) {
                i7 = i0;
            }
            if (kernel_elements <= 8) {
                i8 = i0;
            }

            size_t c = channels;
            do {
                size_t vl = vsetvl_e32m4(c);
                vfloat32m4_t vi0 = vle32_v_f32m4(i0, vl); i0 += vl;
                vfloat32m4_t vi1 = vle32_v_f32m4(i1, vl); i1 += vl;
                vfloat32m4_t vi2 = vle32_v_f32m4(i2, vl); i2 += vl;
                vfloat32m4_t vi3 = vle32_v_f32m4(i3, vl); i3 += vl;
                vfloat32m4_t vi4 = vle32_v_f32m4(i4, vl); i4 += vl;
                vfloat32m4_t vi5 = vle32_v_f32m4(i5, vl); i5 += vl;
                vfloat32m4_t vi6 = vle32_v_f32m4(i6, vl); i6 += vl;
                vfloat32m4_t vi7 = vle32_v_f32m4(i7, vl); i7 += vl;
                vfloat32m4_t vi8 = vle32_v_f32m4(i8, vl); i8 += vl;

                vfloat32m4_t vmax01 = vfmax_vv_f32m4(vi0, vi1, vl);
                vfloat32m4_t vmax23 = vfmax_vv_f32m4(vi2, vi3, vl);
                vfloat32m4_t vmax45 = vfmax_vv_f32m4(vi4, vi5, vl);
                vfloat32m4_t vmax67 = vfmax_vv_f32m4(vi6, vi7, vl);
                vfloat32m4_t vmax018 = vfmax_vv_f32m4(vmax01, vi8, vl);

                vfloat32m4_t vmax2345 = vfmax_vv_f32m4(vmax23, vmax45, vl);
                vfloat32m4_t vmax01678 = vfmax_vv_f32m4(vmax018, vmax67, vl);
                vfloat32m4_t vout = vfmax_vv_f32m4(vmax2345, vmax01678, vl);
                vout = vfmax_vf_f32m4(vout, voutput_min, vl);
                vout = vfmin_vf_f32m4(vout, voutput_max, vl);

                vse32_v_f32m4(o, vout, vl);

                o += vl;
                c -= vl;
            } while (c != 0);
        }

        for (ptrdiff_t k = (ptrdiff_t) kernel_elements - 9; k > 0; k -= 8) {
            const float* i0 = *input++;
            const float* i1 = *input++;
            const float* i2 = *input++;
            const float* i3 = *input++;
            const float* i4 = *input++;
            const float* i5 = *input++;
            const float* i6 = *input++;
            const float* i7 = *input++;
            i0 = (const float*) ((uintptr_t) i0 + input_offset);
            i1 = (const float*) ((uintptr_t) i1 + input_offset);
            i2 = (const float*) ((uintptr_t) i2 + input_offset);
            i3 = (const float*) ((uintptr_t) i3 + input_offset);
            i4 = (const float*) ((uintptr_t) i4 + input_offset);
            i5 = (const float*) ((uintptr_t) i5 + input_offset);
            i6 = (const float*) ((uintptr_t) i6 + input_offset);
            i7 = (const float*) ((uintptr_t) i7 + input_offset);
            if (k < 2) {
                i1 = i0;
            }
            if (k <= 2) {
                i2 = i0;
            }
            if (k < 4) {
                i3 = i0;
            }
            if (k <= 4) {
                i4 = i0;
            }
            if (k < 6) {
                i5 = i0;
            }
            if (k <= 6) {
                i6 = i0;
            }
            if (k < 8) {
                i7 = i0;
            }

            o = output;
            size_t c = channels;
            do {
                size_t vl = vsetvl_e32m4(c);
                vfloat32m4_t vi0 = vle32_v_f32m4(i0, vl); i0 += vl;
                vfloat32m4_t vi1 = vle32_v_f32m4(i1, vl); i1 += vl;
                vfloat32m4_t vi2 = vle32_v_f32m4(i2, vl); i2 += vl;
                vfloat32m4_t vi3 = vle32_v_f32m4(i3, vl); i3 += vl;
                vfloat32m4_t vi4 = vle32_v_f32m4(i4, vl); i4 += vl;
                vfloat32m4_t vi5 = vle32_v_f32m4(i5, vl); i5 += vl;
                vfloat32m4_t vi6 = vle32_v_f32m4(i6, vl); i6 += vl;
                vfloat32m4_t vi7 = vle32_v_f32m4(i7, vl); i7 += vl;
                vfloat32m4_t vi8 = vle32_v_f32m4(o, vl);

                vfloat32m4_t vmax01 = vfmax_vv_f32m4(vi0, vi1, vl);
                vfloat32m4_t vmax23 = vfmax_vv_f32m4(vi2, vi3, vl);
                vfloat32m4_t vmax45 = vfmax_vv_f32m4(vi4, vi5, vl);
                vfloat32m4_t vmax67 = vfmax_vv_f32m4(vi6, vi7, vl);
                vfloat32m4_t vmax018 = vfmax_vv_f32m4(vmax01, vi8, vl);

                vfloat32m4_t vmax2345 = vfmax_vv_f32m4(vmax23, vmax45, vl);
                vfloat32m4_t vmax01678 = vfmax_vv_f32m4(vmax018, vmax67, vl);
                vfloat32m4_t vout = vfmax_vv_f32m4(vmax2345, vmax01678, vl);
                vout = vfmax_vf_f32m4(vout, voutput_min, vl);
                vout = vfmin_vf_f32m4(vout, voutput_max, vl);

                vse32_v_f32m4(o, vout, vl);

                o += vl;
                c -= vl;
            } while (c != 0);
        }
        input = (const float**) ((uintptr_t) input + input_increment);
        output = (float*) ((uintptr_t) o + output_increment);
    } while (--output_pixels != 0);
}

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

    size_t size = batch / sizeof(float);
    do {
        // 动态设置向量长度
        const size_t vl = vsetvl_e32m4(size);

        // 加载输入向量
        vfloat32m4_t vx = vle32_v_f32m4(input, vl);
        input += vl;

        vfloat32m4_t vacc = vfadd_vf_f32m4(vx, vthree, vl);
        vx = vfmul_vf_f32m4(vx, vsixth, vl);

        // 应用最小值约束
        vacc = vfmax_vf_f32m4(vacc, vzero, vl);

        // 应用最大值约束
        vacc = vfmin_vf_f32m4(vacc, vsix, vl);

        vacc = vfmul_vv_f32m4(vacc, vx, vl);

        // 存储结果
        vse32_v_f32m4(output, vacc, vl);
        output += vl;
        size -= vl;
    } while (size > 0);
}

//u8v

void xnn_f32_maxpool_minmax_ukernel_9p8x__rvv_u8v(
        size_t output_pixels,
        size_t kernel_elements,
        size_t channels,
        const float** input,
        size_t input_offset,
        float* output,
        size_t input_increment,
        size_t output_increment,
        const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
    assert(output_pixels != 0);
    assert(kernel_elements != 0);
    assert(channels != 0);

    const float voutput_min = params->scalar.min;
    const float voutput_max = params->scalar.max;
    do {
        float* o = output;
        {
            const float* i0 = *input++;
            const float* i1 = *input++;
            const float* i2 = *input++;
            const float* i3 = *input++;
            const float* i4 = *input++;
            const float* i5 = *input++;
            const float* i6 = *input++;
            const float* i7 = *input++;
            const float* i8 = *input++;
            i0 = (const float*) ((uintptr_t) i0 + input_offset);
            i1 = (const float*) ((uintptr_t) i1 + input_offset);
            i2 = (const float*) ((uintptr_t) i2 + input_offset);
            i3 = (const float*) ((uintptr_t) i3 + input_offset);
            i4 = (const float*) ((uintptr_t) i4 + input_offset);
            i5 = (const float*) ((uintptr_t) i5 + input_offset);
            i6 = (const float*) ((uintptr_t) i6 + input_offset);
            i7 = (const float*) ((uintptr_t) i7 + input_offset);
            i8 = (const float*) ((uintptr_t) i8 + input_offset);
            if (kernel_elements < 2) {
                i1 = i0;
            }
            if (kernel_elements <= 2) {
                i2 = i0;
            }
            if (kernel_elements < 4) {
                i3 = i0;
            }
            if (kernel_elements <= 4) {
                i4 = i0;
            }
            if (kernel_elements < 6) {
                i5 = i0;
            }
            if (kernel_elements <= 6) {
                i6 = i0;
            }
            if (kernel_elements < 8) {
                i7 = i0;
            }
            if (kernel_elements <= 8) {
                i8 = i0;
            }

            size_t c = channels;
            do {
                size_t vl = vsetvl_e32m8(c);
                vfloat32m8_t vi0 = vle32_v_f32m8(i0, vl); i0 += vl;
                vfloat32m8_t vi1 = vle32_v_f32m8(i1, vl); i1 += vl;
                vfloat32m8_t vi2 = vle32_v_f32m8(i2, vl); i2 += vl;
                vfloat32m8_t vi3 = vle32_v_f32m8(i3, vl); i3 += vl;
                vfloat32m8_t vi4 = vle32_v_f32m8(i4, vl); i4 += vl;
                vfloat32m8_t vi5 = vle32_v_f32m8(i5, vl); i5 += vl;
                vfloat32m8_t vi6 = vle32_v_f32m8(i6, vl); i6 += vl;
                vfloat32m8_t vi7 = vle32_v_f32m8(i7, vl); i7 += vl;
                vfloat32m8_t vi8 = vle32_v_f32m8(i8, vl); i8 += vl;

                vfloat32m8_t vmax01 = vfmax_vv_f32m8(vi0, vi1, vl);
                vfloat32m8_t vmax23 = vfmax_vv_f32m8(vi2, vi3, vl);
                vfloat32m8_t vmax45 = vfmax_vv_f32m8(vi4, vi5, vl);
                vfloat32m8_t vmax67 = vfmax_vv_f32m8(vi6, vi7, vl);
                vfloat32m8_t vmax018 = vfmax_vv_f32m8(vmax01, vi8, vl);

                vfloat32m8_t vmax2345 = vfmax_vv_f32m8(vmax23, vmax45, vl);
                vfloat32m8_t vmax01678 = vfmax_vv_f32m8(vmax018, vmax67, vl);
                vfloat32m8_t vout = vfmax_vv_f32m8(vmax2345, vmax01678, vl);
                vout = vfmax_vf_f32m8(vout, voutput_min, vl);
                vout = vfmin_vf_f32m8(vout, voutput_max, vl);

                vse32_v_f32m8(o, vout, vl);

                o += vl;
                c -= vl;
            } while (c != 0);
        }

        for (ptrdiff_t k = (ptrdiff_t) kernel_elements - 9; k > 0; k -= 8) {
            const float* i0 = *input++;
            const float* i1 = *input++;
            const float* i2 = *input++;
            const float* i3 = *input++;
            const float* i4 = *input++;
            const float* i5 = *input++;
            const float* i6 = *input++;
            const float* i7 = *input++;
            i0 = (const float*) ((uintptr_t) i0 + input_offset);
            i1 = (const float*) ((uintptr_t) i1 + input_offset);
            i2 = (const float*) ((uintptr_t) i2 + input_offset);
            i3 = (const float*) ((uintptr_t) i3 + input_offset);
            i4 = (const float*) ((uintptr_t) i4 + input_offset);
            i5 = (const float*) ((uintptr_t) i5 + input_offset);
            i6 = (const float*) ((uintptr_t) i6 + input_offset);
            i7 = (const float*) ((uintptr_t) i7 + input_offset);
            if (k < 2) {
                i1 = i0;
            }
            if (k <= 2) {
                i2 = i0;
            }
            if (k < 4) {
                i3 = i0;
            }
            if (k <= 4) {
                i4 = i0;
            }
            if (k < 6) {
                i5 = i0;
            }
            if (k <= 6) {
                i6 = i0;
            }
            if (k < 8) {
                i7 = i0;
            }

            o = output;
            size_t c = channels;
            do {
                size_t vl = vsetvl_e32m8(c);
                vfloat32m8_t vi0 = vle32_v_f32m8(i0, vl); i0 += vl;
                vfloat32m8_t vi1 = vle32_v_f32m8(i1, vl); i1 += vl;
                vfloat32m8_t vi2 = vle32_v_f32m8(i2, vl); i2 += vl;
                vfloat32m8_t vi3 = vle32_v_f32m8(i3, vl); i3 += vl;
                vfloat32m8_t vi4 = vle32_v_f32m8(i4, vl); i4 += vl;
                vfloat32m8_t vi5 = vle32_v_f32m8(i5, vl); i5 += vl;
                vfloat32m8_t vi6 = vle32_v_f32m8(i6, vl); i6 += vl;
                vfloat32m8_t vi7 = vle32_v_f32m8(i7, vl); i7 += vl;
                vfloat32m8_t vi8 = vle32_v_f32m8(o, vl);

                vfloat32m8_t vmax01 = vfmax_vv_f32m8(vi0, vi1, vl);
                vfloat32m8_t vmax23 = vfmax_vv_f32m8(vi2, vi3, vl);
                vfloat32m8_t vmax45 = vfmax_vv_f32m8(vi4, vi5, vl);
                vfloat32m8_t vmax67 = vfmax_vv_f32m8(vi6, vi7, vl);
                vfloat32m8_t vmax018 = vfmax_vv_f32m8(vmax01, vi8, vl);

                vfloat32m8_t vmax2345 = vfmax_vv_f32m8(vmax23, vmax45, vl);
                vfloat32m8_t vmax01678 = vfmax_vv_f32m8(vmax018, vmax67, vl);
                vfloat32m8_t vout = vfmax_vv_f32m8(vmax2345, vmax01678, vl);
                vout = vfmax_vf_f32m8(vout, voutput_min, vl);
                vout = vfmin_vf_f32m8(vout, voutput_max, vl);

                vse32_v_f32m8(o, vout, vl);

                o += vl;
                c -= vl;
            } while (c != 0);
        }
        input = (const float**) ((uintptr_t) input + input_increment);
        output = (float*) ((uintptr_t) o + output_increment);
    } while (--output_pixels != 0);
}

void xnn_f32_vhswish_ukernel__rvv_u8v(
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

    size_t size = batch / sizeof(float);
    do {
        // 动态设置向量长度
        const size_t vl = vsetvl_e32m8(size);

        // 加载输入向量
        vfloat32m8_t vx = vle32_v_f32m8(input, vl);
        input += vl;

        vfloat32m8_t vacc = vfadd_vf_f32m8(vx, vthree, vl);
        vx = vfmul_vf_f32m8(vx, vsixth, vl);

        // 应用最小值约束
        vacc = vfmax_vf_f32m8(vacc, vzero, vl);

        // 应用最大值约束
        vacc = vfmin_vf_f32m8(vacc, vsix, vl);

        vacc = vfmul_vv_f32m8(vacc, vx, vl);

        // 存储结果
        vse32_v_f32m8(output, vacc, vl);
        output += vl;
        size -= vl;
    } while (size > 0);
}


// void xnn_f16_gemm_ukernel_1x16__rvv_u2v(
//         size_t mr,
//         size_t nc,
//         size_t kc,
//         const void* restrict a,
//         size_t a_stride,
//         const void* restrict w,
//         void* restrict c,
//         size_t cm_stride,
//         size_t cn_stride,
//         const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)])
// {
//     assert(mr != 0);
//     assert(mr <= 1);
//     assert(nc != 0);
//     assert(kc != 0);
//     assert(kc % sizeof(__fp16) == 0);
//     assert(a != NULL);
//     assert(w != NULL);
//     assert(c != NULL);

//     const __fp16* a0 = a;
//     __fp16* c0 = c;
//     __fp16* hw = (__fp16*)w;

//     size_t kcl = kc / sizeof(__fp16);

//     do {
//         size_t vl = vsetvl_e16m2(nc);
//         vfloat16m2_t vacc = vle16_v_f16m2(hw, 16);
//         hw += 16;
//         for(size_t k = 0; k < kcl ; k++){
//             vfloat16m2_t vw = vle16_v_f16m2(hw, 16);
//             hw += 16;
//             vacc = vfmacc_vf_f16m2(vacc, *a0, vw, 16);
//             a0++;
//         }
//         vse16_v_f16m2(c0, vacc, vl);
//         if(nc >= 16){
//             c0 = (__fp16*) ((uintptr_t) c0 + cn_stride);
//             a0 = (const void*) ((uintptr_t) a0 - kc);
//         }
//         nc -= vl;
//     } while (nc != 0);
// }

// void xnn_f16_gemm_ukernel_4x16__rvv_u2v(
//         size_t mr,
//         size_t nc,
//         size_t kc,
//         const void* restrict a,
//         size_t a_stride,
//         const void* restrict w,
//         void* restrict c,
//         size_t cm_stride,
//         size_t cn_stride,
//         const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)])
// {
//     assert(mr != 0);
//     assert(mr <= 4); // max process 1 row
//     assert(nc != 0);
//     assert(kc != 0);
//     assert(kc % sizeof(__fp16) == 0);
//     assert(a != NULL);
//     assert(w != NULL);
//     assert(c != NULL);

//     const __fp16* a0 = (__fp16*)a;
//     __fp16* c0 = c;
//     const __fp16* a1 = (const __fp16*) ((uintptr_t) a0 + a_stride);
//     __fp16* c1 = (__fp16*) ((uintptr_t) c0 + cm_stride);
//     if XNN_UNPREDICTABLE(mr < 2) {
//         a1 = a0;
//         c1 = c0;
//     }
//     const __fp16* a2 = (const __fp16*) ((uintptr_t) a1 + a_stride);
//     __fp16* c2 = (__fp16*) ((uintptr_t) c1 + cm_stride);
//     if XNN_UNPREDICTABLE(mr <= 2) {
//         a2 = a1;
//         c2 = c1;
//     }
//     const __fp16* a3 = (const __fp16*) ((uintptr_t) a2 + a_stride);
//     __fp16* c3 = (__fp16*) ((uintptr_t) c2 + cm_stride);
//     if XNN_UNPREDICTABLE(mr != 4) {
//         a3 = a2;
//         c3 = c2;
//     }

//     __fp16* hw = (__fp16*)w;


//     size_t kcl = kc / sizeof(__fp16);

//     do {
//         size_t vl = vsetvl_e16m2(nc); // vector length
//         vfloat16m2_t vacc0 = vle16_v_f16m2(hw, 16); // 1st row count
//         vfloat16m2_t vacc1 = vle16_v_f16m2(hw, 16); // 1st row count
//         vfloat16m2_t vacc2 = vle16_v_f16m2(hw, 16); // 1st row count
//         vfloat16m2_t vacc3 = vle16_v_f16m2(hw, 16); // 1st row count
//         hw += 16;
//         for(size_t k = 0; k < kcl ; k++){
//             vfloat16m2_t vw = vle16_v_f16m2(hw, 16);
//             hw += 16;
//             vacc0 = vfmacc_vf_f16m2(vacc0, *a0, vw, 16); // update 1st row count
//             vacc1 = vfmacc_vf_f16m2(vacc1, *a1, vw, 16); // update 1st row count
//             vacc2 = vfmacc_vf_f16m2(vacc2, *a2, vw, 16); // update 1st row count
//             vacc3 = vfmacc_vf_f16m2(vacc3, *a3, vw, 16); // update 1st row count
//             a0++;
//             a1++;
//             a2++;
//             a3++;
//         }
//         vse16_v_f16m2(c0, vacc0, vl); // store 1st row result
//         vse16_v_f16m2(c1, vacc1, vl); // store 1st row result
//         vse16_v_f16m2(c2, vacc2, vl); // store 1st row result
//         vse16_v_f16m2(c3, vacc3, vl); // store 1st row result
//         if(nc >= 16){
//             c0 = (__fp16*) ((uintptr_t) c0 + cn_stride); // update 1st row matrix C pointer
//             c1 = (__fp16*) ((uintptr_t) c1 + cn_stride); // update 1st row matrix C pointer
//             c2 = (__fp16*) ((uintptr_t) c2 + cn_stride); // update 1st row matrix C pointer
//             c3 = (__fp16*) ((uintptr_t) c3 + cn_stride); // update 1st row matrix C pointer
//             a0 = (const void*) ((uintptr_t) a0 - kc); // update 1st row matrix A pointer
//             a1 = (const void*) ((uintptr_t) a1 - kc); // update 1st row matrix A pointer
//             a2 = (const void*) ((uintptr_t) a2 - kc); // update 1st row matrix A pointer
//             a3 = (const void*) ((uintptr_t) a3 - kc); // update 1st row matrix A pointer
//         }
//         nc -= vl;
//     } while (nc != 0);
// }

// void xnn_f16_igemm_ukernel_1x16__rvv_u2v(
//         size_t mr,
//         size_t nc,
//         size_t kc,
//         size_t ks,
//         const void** restrict a,
//         const void* restrict w,
//         void* restrict c,
//         size_t cm_stride,
//         size_t cn_stride,
//         size_t a_offset,
//         const void* zero,
//         const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)])
// {
//     assert(mr != 0);
//     assert(mr <= 1);
//     assert(nc != 0);
//     assert(kc != 0);
//     assert(kc % sizeof(float) == 0);
//     assert(ks != 0);
//     assert(ks % (1 * sizeof(void*)) == 0);
//     assert(a_offset % sizeof(float) == 0);
//     assert(a != NULL);
//     assert(w != NULL);
//     assert(c != NULL);

//     float* c0 = c;

//     __fp16* hw = (__fp16*)w;

//     do {
//         size_t vl = vsetvl_e16m2(nc); // vector length
//         vfloat16m2_t vacc0 = vle16_v_f16m2(hw, 16); // 1st row count
//         hw += 16;

//         size_t p = ks;
//         size_t kcl = kc / sizeof(float);
//         do {
//             const float* restrict a0 = a[0];
//             assert(a0 != NULL);
//             if XNN_UNPREDICTABLE(a0 != zero) {
//                 a0 = (const float*) ((uintptr_t) a0 + a_offset);
//             }
//             a += 1;

//             size_t k = kc;
//             for(size_t k = 0; k < kcl ; k++){
//                 vfloat16m2_t vw = vle16_v_f16m2(hw, 16);
//                 hw += 16;
//                 vacc0 = vfmacc_vf_f16m2(vacc0, *a0, vw, 16); // update 1st row count
//                 a0++;
//             }
//             p -= 1 * sizeof(void*);
//         } while (p != 0);
//         vse16_v_f16m2(c0, vacc0, vl); // store 1st row result

//         if XNN_LIKELY(nc >= 16) {
//             c0 = (float*) ((uintptr_t) c0 + cn_stride);
//             a = (const float**restrict) ((uintptr_t) a - ks);
//         }
//         nc -= vl;
//     } while (nc != 0);
// }



// void xnn_f16_igemm_ukernel_4x16__rvv_u2v(
//         size_t mr,
//         size_t nc,
//         size_t kc,
//         size_t ks,
//         const void** restrict a,
//         const void* restrict w,
//         void* restrict c,
//         size_t cm_stride,
//         size_t cn_stride,
//         size_t a_offset,const void* zero,//todo
//         const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)])
// {
//     assert(mr != 0);
//     assert(mr <= 4);
//     assert(nc != 0);
//     assert(kc != 0);
//     assert(kc % sizeof(__fp16) == 0);
//     assert(ks != 0);
//     assert(ks % (4 * sizeof(void*)) == 0);
//     assert(a_offset % sizeof(__fp16) == 0);
//     assert(a != NULL);
//     assert(w != NULL);
//     assert(c != NULL);

//     __fp16* c0 = c;
//     __fp16* c1 = (__fp16*) ((uintptr_t) c0 + cm_stride);
//     if XNN_UNPREDICTABLE(mr < 2) {
//         c1 = c0;
//     }
//     __fp16* c2 = (__fp16*) ((uintptr_t) c1 + cm_stride);
//     if XNN_UNPREDICTABLE(mr <= 2) {
//         c2 = c1;
//     }
//     __fp16* c3 = (__fp16*) ((uintptr_t) c2 + cm_stride);
//     if XNN_UNPREDICTABLE(mr != 4) {
//         c3 = c2;
//     }

//     __fp16* hw = (__fp16*)w;

//     do {
//         size_t vl = vsetvl_e16m2(nc); // vector length
//         vfloat16m2_t vacc0 = vle16_v_f16m2(hw, 16); // 1st row count
//         vfloat16m2_t vacc1 = vle16_v_f16m2(hw, 16); // 1st row count
//         vfloat16m2_t vacc2 = vle16_v_f16m2(hw, 16); // 1st row count
//         vfloat16m2_t vacc3 = vle16_v_f16m2(hw, 16); // 1st row count
//         hw += 16;

//         size_t p = ks;
//         size_t kcl = kc / sizeof(__fp16);
//         do {
//             const __fp16* restrict a0 = a[0];
//             assert(a0 != NULL);
//             if XNN_UNPREDICTABLE(a0 != zero) {
//                 a0 = (const __fp16*) ((uintptr_t) a0 + a_offset);
//             }
//             const __fp16* restrict a1 = a[1];
//             assert(a1 != NULL);
//             if XNN_UNPREDICTABLE(a1 != zero) {
//                 a1 = (const __fp16*) ((uintptr_t) a1 + a_offset);
//             }
//             const __fp16* restrict a2 = a[2];
//             assert(a2 != NULL);
//             if XNN_UNPREDICTABLE(a2 != zero) {
//                 a2 = (const __fp16*) ((uintptr_t) a2 + a_offset);
//             }
//             const __fp16* restrict a3 = a[3];
//             assert(a3 != NULL);
//             if XNN_UNPREDICTABLE(a3 != zero) {
//                 a3 = (const __fp16*) ((uintptr_t) a3 + a_offset);
//             }
//             a += 4;

//             size_t k = kc;
//             for(size_t k = 0; k < kcl ; k++){
//                 vfloat16m2_t vw = vle16_v_f16m2(hw, 16);
//                 hw += 16;
//                 vacc0 = vfmacc_vf_f16m2(vacc0, *a0, vw, 16); // update 1st row count
//                 vacc1 = vfmacc_vf_f16m2(vacc1, *a1, vw, 16); // update 1st row count
//                 vacc2 = vfmacc_vf_f16m2(vacc2, *a2, vw, 16); // update 1st row count
//                 vacc3 = vfmacc_vf_f16m2(vacc3, *a3, vw, 16); // update 1st row count
//                 a0++;
//                 a1++;
//                 a2++;
//                 a3++;
//             }
//             p -= 4 * sizeof(void*);//todo
//         } while (p != 0);
//         vse16_v_f16m2(c0, vacc0, vl); // store 1st row result
//         vse16_v_f16m2(c1, vacc1, vl); // store 1st row result
//         vse16_v_f16m2(c2, vacc2, vl); // store 1st row result
//         vse16_v_f16m2(c3, vacc3, vl); // store 1st row result

//         if XNN_LIKELY(nc >= 16) {
//             c3 = (__fp16*) ((uintptr_t) c3 + cn_stride);
//             c2 = (__fp16*) ((uintptr_t) c2 + cn_stride);
//             c1 = (__fp16*) ((uintptr_t) c1 + cn_stride);
//             c0 = (__fp16*) ((uintptr_t) c0 + cn_stride);

//             a = (const void**restrict) ((uintptr_t) a - ks);
//         }
//         nc -= vl;
//     } while (nc != 0);
// }

// void xnn_f16_gemm_minmax_ukernel_1x16__rvv_u2v(
//         size_t mr,
//         size_t nc,
//         size_t kc,
//         const void* restrict a,
//         size_t a_stride,
//         const void* restrict w,
//         void* restrict c,
//         size_t cm_stride,
//         size_t cn_stride,
//         const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
// {
//     assert(mr != 0);
//     assert(mr <= 1);
//     assert(nc != 0);
//     assert(kc != 0);
//     assert(kc % sizeof(__fp16) == 0);
//     assert(a != NULL);
//     assert(w != NULL);
//     assert(c != NULL);

//     const __fp16* a0 = a;
//     __fp16* c0 = c;
//     size_t kcl = kc / sizeof(__fp16);

//     __fp16 vmin, vmax;
//     memcpy(&vmin, &params->fp16arith.min, sizeof(vmin));
//     memcpy(&vmax, &params->fp16arith.max, sizeof(vmax));

//     __fp16* hw = (__fp16*)w;

//     do {
//         size_t vl = vsetvl_e16m2(nc);
//         vfloat16m2_t vacc = vle16_v_f16m2(hw, 16);
//         hw += 16;
//         for(size_t k = 0; k < kcl ; k++){
//             vfloat16m2_t vw = vle16_v_f16m2(hw, 16);
//             hw += 16;
//             vacc = vfmacc_vf_f16m2(vacc, *a0, vw, 16);
//             a0++;
//         }
//         vacc = vfmax_vf_f16m2(vacc, vmin, vl);
//         vacc = vfmin_vf_f16m2(vacc, vmax, vl);
//         vse16_v_f16m2(c0, vacc, vl);
//         if(nc >= 16){
//             c0 = (__fp16*) ((uintptr_t) c0 + cn_stride);
//             a0 = (const void*) ((uintptr_t) a0 - kc);
//         }
//         nc -= vl;
//     } while (nc != 0);
// }

// void xnn_f16_gemm_minmax_ukernel_4x16__rvv_u2v(
//         size_t mr,
//         size_t nc,
//         size_t kc,
//         const void* restrict a,
//         size_t a_stride,
//         const void* restrict w,
//         void* restrict c,
//         size_t cm_stride,
//         size_t cn_stride,
//         const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
// {
//     assert(mr != 0);
//     assert(mr <= 4); // max process 1 row
//     assert(nc != 0);
//     assert(kc != 0);
//     assert(kc % sizeof(__fp16) == 0);
//     assert(a != NULL);
//     assert(w != NULL);
//     assert(c != NULL);

//     const __fp16* a0 = a;
//     __fp16* c0 = c;
//     const __fp16* a1 = (const __fp16*) ((uintptr_t) a0 + a_stride);
//     __fp16* c1 = (__fp16*) ((uintptr_t) c0 + cm_stride);
//     if XNN_UNPREDICTABLE(mr < 2) {
//         a1 = a0;
//         c1 = c0;
//     }
//     const __fp16* a2 = (const __fp16*) ((uintptr_t) a1 + a_stride);
//     __fp16* c2 = (__fp16*) ((uintptr_t) c1 + cm_stride);
//     if XNN_UNPREDICTABLE(mr <= 2) {
//         a2 = a1;
//         c2 = c1;
//     }
//     const __fp16* a3 = (const __fp16*) ((uintptr_t) a2 + a_stride);
//     __fp16* c3 = (__fp16*) ((uintptr_t) c2 + cm_stride);
//     if XNN_UNPREDICTABLE(mr != 4) {
//         a3 = a2;
//         c3 = c2;
//     }

//     __fp16 vmin, vmax;
//     memcpy(&vmin, &params->fp16arith.min, sizeof(vmin));
//     memcpy(&vmax, &params->fp16arith.max, sizeof(vmax));

//     __fp16* hw = (__fp16*)w;

//     size_t kcl = kc / sizeof(__fp16);

//     do {
//         size_t vl = vsetvl_e16m2(nc); // vector length
//         vfloat16m2_t vacc0 = vle16_v_f16m2(hw, 16); // 1st row count
//         vfloat16m2_t vacc1 = vle16_v_f16m2(hw, 16); // 1st row count
//         vfloat16m2_t vacc2 = vle16_v_f16m2(hw, 16); // 1st row count
//         vfloat16m2_t vacc3 = vle16_v_f16m2(hw, 16); // 1st row count
//         hw += 16;
//         for(size_t k = 0; k < kcl ; k++){
//             vfloat16m2_t vw = vle16_v_f16m2(hw, 16);
//             hw += 16;
//             vacc0 = vfmacc_vf_f16m2(vacc0, *a0, vw, 16); // update 1st row count
//             vacc1 = vfmacc_vf_f16m2(vacc1, *a1, vw, 16); // update 1st row count
//             vacc2 = vfmacc_vf_f16m2(vacc2, *a2, vw, 16); // update 1st row count
//             vacc3 = vfmacc_vf_f16m2(vacc3, *a3, vw, 16); // update 1st row count
//             a0++;
//             a1++;
//             a2++;
//             a3++;
//         }
//         vacc0 = vfmax_vf_f16m2(vacc0, vmin, vl);
//         vacc1 = vfmax_vf_f16m2(vacc1, vmin, vl);
//         vacc2 = vfmax_vf_f16m2(vacc2, vmin, vl);
//         vacc3 = vfmax_vf_f16m2(vacc3, vmin, vl);

//         vacc0 = vfmin_vf_f16m2(vacc0, vmax, vl);
//         vacc1 = vfmin_vf_f16m2(vacc1, vmax, vl);
//         vacc2 = vfmin_vf_f16m2(vacc2, vmax, vl);
//         vacc3 = vfmin_vf_f16m2(vacc3, vmax, vl);

//         vse16_v_f16m2(c0, vacc0, vl); // store 1st row result
//         vse16_v_f16m2(c1, vacc1, vl); // store 1st row result
//         vse16_v_f16m2(c2, vacc2, vl); // store 1st row result
//         vse16_v_f16m2(c3, vacc3, vl); // store 1st row result
//         if(nc >= 16){
//             c0 = (__fp16*) ((uintptr_t) c0 + cn_stride); // update 1st row matrix C pointer
//             c1 = (__fp16*) ((uintptr_t) c1 + cn_stride); // update 1st row matrix C pointer
//             c2 = (__fp16*) ((uintptr_t) c2 + cn_stride); // update 1st row matrix C pointer
//             c3 = (__fp16*) ((uintptr_t) c3 + cn_stride); // update 1st row matrix C pointer
//             a0 = (const void*) ((uintptr_t) a0 - kc); // update 1st row matrix A pointer
//             a1 = (const void*) ((uintptr_t) a1 - kc); // update 1st row matrix A pointer
//             a2 = (const void*) ((uintptr_t) a2 - kc); // update 1st row matrix A pointer
//             a3 = (const void*) ((uintptr_t) a3 - kc); // update 1st row matrix A pointer
//         }
//         nc -= vl;
//     } while (nc != 0);
// }

// void xnn_f16_igemm_minmax_ukernel_1x16__rvv_u2v(
//         size_t mr,
//         size_t nc,
//         size_t kc,
//         size_t ks,
//         const void** restrict a,
//         const void* restrict w,
//         void* restrict c,
//         size_t cm_stride,
//         size_t cn_stride,
//         size_t a_offset,
//         const void* zero,
//         const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
// {
//     assert(mr != 0);
//     assert(mr <= 1);
//     assert(nc != 0);
//     assert(kc != 0);
//     assert(kc % sizeof(__fp16) == 0);
//     assert(ks != 0);
//     assert(ks % (1 * sizeof(void*)) == 0);
//     assert(a_offset % sizeof(__fp16) == 0);
//     assert(a != NULL);
//     assert(w != NULL);
//     assert(c != NULL);

//     __fp16* c0 = c;

//     __fp16 vmin, vmax;
//     memcpy(&vmin, &params->fp16arith.min, sizeof(vmin));
//     memcpy(&vmax, &params->fp16arith.max, sizeof(vmax));

//     __fp16* hw = (__fp16*)w;


//     do {
//         size_t vl = vsetvl_e16m2(nc); // vector length
//         vfloat16m2_t vacc0 = vle16_v_f16m2(hw, 16); // 1st row count
//         hw += 16;

//         size_t p = ks;
//         size_t kcl = kc / sizeof(__fp16);
//         do {
//             const __fp16* restrict a0 = a[0];
//             assert(a0 != NULL);
//             if XNN_UNPREDICTABLE(a0 != zero) {
//                 a0 = (const __fp16*) ((uintptr_t) a0 + a_offset);
//             }
//             a += 1;

//             size_t k = kc;
//             for(size_t k = 0; k < kcl ; k++){
//                 vfloat16m2_t vw = vle16_v_f16m2(hw, 16);
//                 hw += 16;
//                 vacc0 = vfmacc_vf_f16m2(vacc0, *a0, vw, 16); // update 1st row count
//                 a0++;
//             }
//             p -= 1 * sizeof(void*);
//         } while (p != 0);
//         vacc0 = vfmax_vf_f16m2(vacc0, vmin, vl);
//         vacc0 = vfmin_vf_f16m2(vacc0, vmax, vl);
//         vse16_v_f16m2(c0, vacc0, vl); // store 1st row result

//         if XNN_LIKELY(nc >= 16) {
//             c0 = (__fp16*) ((uintptr_t) c0 + cn_stride);
//             a = (const void**restrict) ((uintptr_t) a - ks);
//         }
//         nc -= vl;
//     } while (nc != 0);
// }

// void xnn_f16_igemm_minmax_ukernel_4x16__rvv_u2v(
//         size_t mr,
//         size_t nc,
//         size_t kc,
//         size_t ks,
//         const void** restrict a,
//         const void* restrict w,
//         void* restrict c,
//         size_t cm_stride,
//         size_t cn_stride,
//         size_t a_offset,const void* zero,//todo
//         const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
// {
//     assert(mr != 0);
//     assert(mr <= 4);
//     assert(nc != 0);
//     assert(kc != 0);
//     assert(kc % sizeof(__fp16) == 0);
//     assert(ks != 0);
//     assert(ks % (4 * sizeof(void*)) == 0);
//     assert(a_offset % sizeof(__fp16) == 0);
//     assert(a != NULL);
//     assert(w != NULL);
//     assert(c != NULL);

//     __fp16* c0 = c;
//     __fp16* c1 = (__fp16*) ((uintptr_t) c0 + cm_stride);
//     if XNN_UNPREDICTABLE(mr < 2) {
//         c1 = c0;
//     }
//     __fp16* c2 = (__fp16*) ((uintptr_t) c1 + cm_stride);
//     if XNN_UNPREDICTABLE(mr <= 2) {
//         c2 = c1;
//     }
//     __fp16* c3 = (__fp16*) ((uintptr_t) c2 + cm_stride);
//     if XNN_UNPREDICTABLE(mr != 4) {
//         c3 = c2;
//     }

//     __fp16 vmin, vmax;
//     memcpy(&vmin, &params->fp16arith.min, sizeof(vmin));
//     memcpy(&vmax, &params->fp16arith.max, sizeof(vmax));

//     __fp16* hw = (__fp16*)w;


//     do {
//         size_t vl = vsetvl_e16m2(nc); // vector length
//         vfloat16m2_t vacc0 = vle16_v_f16m2(hw, 16); // 1st row count
//         vfloat16m2_t vacc1 = vle16_v_f16m2(hw, 16); // 1st row count
//         vfloat16m2_t vacc2 = vle16_v_f16m2(hw, 16); // 1st row count
//         vfloat16m2_t vacc3 = vle16_v_f16m2(hw, 16); // 1st row count

//         hw += 16;

//         size_t p = ks;
//         size_t kcl = kc / sizeof(__fp16);
//         do {
//             const __fp16* restrict a0 = a[0];
//             assert(a0 != NULL);
//             if XNN_UNPREDICTABLE(a0 != zero) {
//                 a0 = (const __fp16*) ((uintptr_t) a0 + a_offset);
//             }
//             const __fp16* restrict a1 = a[1];
//             assert(a1 != NULL);
//             if XNN_UNPREDICTABLE(a1 != zero) {
//                 a1 = (const __fp16*) ((uintptr_t) a1 + a_offset);
//             }
//             const __fp16* restrict a2 = a[2];
//             assert(a2 != NULL);
//             if XNN_UNPREDICTABLE(a2 != zero) {
//                 a2 = (const __fp16*) ((uintptr_t) a2 + a_offset);
//             }
//             const __fp16* restrict a3 = a[3];
//             assert(a3 != NULL);
//             if XNN_UNPREDICTABLE(a3 != zero) {
//                 a3 = (const __fp16*) ((uintptr_t) a3 + a_offset);
//             }
//             a += 4;

//             size_t k = kc;
//             for(size_t k = 0; k < kcl ; k++){
//                 vfloat16m2_t vw = vle16_v_f16m2(hw, 16);
//                 hw += 16;
//                 vacc0 = vfmacc_vf_f16m2(vacc0, *a0, vw, 16); // update 1st row count
//                 vacc1 = vfmacc_vf_f16m2(vacc1, *a1, vw, 16); // update 1st row count
//                 vacc2 = vfmacc_vf_f16m2(vacc2, *a2, vw, 16); // update 1st row count
//                 vacc3 = vfmacc_vf_f16m2(vacc3, *a3, vw, 16); // update 1st row count
//                 a0++;
//                 a1++;
//                 a2++;
//                 a3++;
//             }
//             p -= 4 * sizeof(void*);//todo
//         } while (p != 0);

//         vacc0 = vfmax_vf_f16m2(vacc0, vmin, vl);
//         vacc1 = vfmax_vf_f16m2(vacc1, vmin, vl);
//         vacc2 = vfmax_vf_f16m2(vacc2, vmin, vl);
//         vacc3 = vfmax_vf_f16m2(vacc3, vmin, vl);

//         vacc0 = vfmin_vf_f16m2(vacc0, vmax, vl);
//         vacc1 = vfmin_vf_f16m2(vacc1, vmax, vl);
//         vacc2 = vfmin_vf_f16m2(vacc2, vmax, vl);
//         vacc3 = vfmin_vf_f16m2(vacc3, vmax, vl);

//         vse16_v_f16m2(c0, vacc0, vl); // store 1st row result
//         vse16_v_f16m2(c1, vacc1, vl); // store 1st row result
//         vse16_v_f16m2(c2, vacc2, vl); // store 1st row result
//         vse16_v_f16m2(c3, vacc3, vl); // store 1st row result

//         if XNN_LIKELY(nc >= 16) {
//             c3 = (__fp16*) ((uintptr_t) c3 + cn_stride);
//             c2 = (__fp16*) ((uintptr_t) c2 + cn_stride);
//             c1 = (__fp16*) ((uintptr_t) c1 + cn_stride);
//             c0 = (__fp16*) ((uintptr_t) c0 + cn_stride);

//             a = (const void**restrict) ((uintptr_t) a - ks);
//         }
//         nc -= vl;
//     } while (nc != 0);
// }

// void xnn_f16_maxpool_minmax_ukernel_9p8x__rvv_u2v(
//         size_t output_pixels,
//         size_t kernel_elements,
//         size_t channels,
//         const void** input,
//         size_t input_offset,
//         void* output,
//         size_t input_increment,
//         size_t output_increment,
//         const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
// {
//     assert(output_pixels != 0);
//     assert(kernel_elements != 0);
//     assert(channels != 0);

//     __fp16 voutput_min, voutput_max;
//     memcpy(&voutput_min, &params->fp16arith.min, sizeof(voutput_min));
//     memcpy(&voutput_max, &params->fp16arith.max, sizeof(voutput_max));

//     do {
//         __fp16* o = output;
//         {
//             const __fp16* i0 = *input++;
//             const __fp16* i1 = *input++;
//             const __fp16* i2 = *input++;
//             const __fp16* i3 = *input++;
//             const __fp16* i4 = *input++;
//             const __fp16* i5 = *input++;
//             const __fp16* i6 = *input++;
//             const __fp16* i7 = *input++;
//             const __fp16* i8 = *input++;
//             i0 = (const __fp16*) ((uintptr_t) i0 + input_offset);
//             i1 = (const __fp16*) ((uintptr_t) i1 + input_offset);
//             i2 = (const __fp16*) ((uintptr_t) i2 + input_offset);
//             i3 = (const __fp16*) ((uintptr_t) i3 + input_offset);
//             i4 = (const __fp16*) ((uintptr_t) i4 + input_offset);
//             i5 = (const __fp16*) ((uintptr_t) i5 + input_offset);
//             i6 = (const __fp16*) ((uintptr_t) i6 + input_offset);
//             i7 = (const __fp16*) ((uintptr_t) i7 + input_offset);
//             i8 = (const __fp16*) ((uintptr_t) i8 + input_offset);
//             if (kernel_elements < 2) {
//                 i1 = i0;
//             }
//             if (kernel_elements <= 2) {
//                 i2 = i0;
//             }
//             if (kernel_elements < 4) {
//                 i3 = i0;
//             }
//             if (kernel_elements <= 4) {
//                 i4 = i0;
//             }
//             if (kernel_elements < 6) {
//                 i5 = i0;
//             }
//             if (kernel_elements <= 6) {
//                 i6 = i0;
//             }
//             if (kernel_elements < 8) {
//                 i7 = i0;
//             }
//             if (kernel_elements <= 8) {
//                 i8 = i0;
//             }

//             size_t c = channels;
//             do {
//                 size_t vl = vsetvl_e16m2(c);
//                 vfloat16m2_t vi0 = vle16_v_f16m2(i0, vl); i0 += vl;
//                 vfloat16m2_t vi1 = vle16_v_f16m2(i1, vl); i1 += vl;
//                 vfloat16m2_t vi2 = vle16_v_f16m2(i2, vl); i2 += vl;
//                 vfloat16m2_t vi3 = vle16_v_f16m2(i3, vl); i3 += vl;
//                 vfloat16m2_t vi4 = vle16_v_f16m2(i4, vl); i4 += vl;
//                 vfloat16m2_t vi5 = vle16_v_f16m2(i5, vl); i5 += vl;
//                 vfloat16m2_t vi6 = vle16_v_f16m2(i6, vl); i6 += vl;
//                 vfloat16m2_t vi7 = vle16_v_f16m2(i7, vl); i7 += vl;
//                 vfloat16m2_t vi8 = vle16_v_f16m2(i8, vl); i8 += vl;

//                 vfloat16m2_t vmax01 = vfmax_vv_f16m2(vi0, vi1, vl);
//                 vfloat16m2_t vmax23 = vfmax_vv_f16m2(vi2, vi3, vl);
//                 vfloat16m2_t vmax45 = vfmax_vv_f16m2(vi4, vi5, vl);
//                 vfloat16m2_t vmax67 = vfmax_vv_f16m2(vi6, vi7, vl);
//                 vfloat16m2_t vmax018 = vfmax_vv_f16m2(vmax01, vi8, vl);

//                 vfloat16m2_t vmax2345 = vfmax_vv_f16m2(vmax23, vmax45, vl);
//                 vfloat16m2_t vmax01678 = vfmax_vv_f16m2(vmax018, vmax67, vl);
//                 vfloat16m2_t vout = vfmax_vv_f16m2(vmax2345, vmax01678, vl);
//                 vout = vfmax_vf_f16m2(vout, voutput_min, vl);
//                 vout = vfmin_vf_f16m2(vout, voutput_max, vl);

//                 vse16_v_f16m2(o, vout, vl);

//                 o += vl;
//                 c -= vl;
//             } while (c != 0);
//         }

//         for (ptrdiff_t k = (ptrdiff_t) kernel_elements - 9; k > 0; k -= 8) {
//             const __fp16* i0 = *input++;
//             const __fp16* i1 = *input++;
//             const __fp16* i2 = *input++;
//             const __fp16* i3 = *input++;
//             const __fp16* i4 = *input++;
//             const __fp16* i5 = *input++;
//             const __fp16* i6 = *input++;
//             const __fp16* i7 = *input++;
//             i0 = (const __fp16*) ((uintptr_t) i0 + input_offset);
//             i1 = (const __fp16*) ((uintptr_t) i1 + input_offset);
//             i2 = (const __fp16*) ((uintptr_t) i2 + input_offset);
//             i3 = (const __fp16*) ((uintptr_t) i3 + input_offset);
//             i4 = (const __fp16*) ((uintptr_t) i4 + input_offset);
//             i5 = (const __fp16*) ((uintptr_t) i5 + input_offset);
//             i6 = (const __fp16*) ((uintptr_t) i6 + input_offset);
//             i7 = (const __fp16*) ((uintptr_t) i7 + input_offset);
//             if (k < 2) {
//                 i1 = i0;
//             }
//             if (k <= 2) {
//                 i2 = i0;
//             }
//             if (k < 4) {
//                 i3 = i0;
//             }
//             if (k <= 4) {
//                 i4 = i0;
//             }
//             if (k < 6) {
//                 i5 = i0;
//             }
//             if (k <= 6) {
//                 i6 = i0;
//             }
//             if (k < 8) {
//                 i7 = i0;
//             }

//             o = output;
//             size_t c = channels;
//             do {
//                 size_t vl = vsetvl_e16m2(c);
//                 vfloat16m2_t vi0 = vle16_v_f16m2(i0, vl); i0 += vl;
//                 vfloat16m2_t vi1 = vle16_v_f16m2(i1, vl); i1 += vl;
//                 vfloat16m2_t vi2 = vle16_v_f16m2(i2, vl); i2 += vl;
//                 vfloat16m2_t vi3 = vle16_v_f16m2(i3, vl); i3 += vl;
//                 vfloat16m2_t vi4 = vle16_v_f16m2(i4, vl); i4 += vl;
//                 vfloat16m2_t vi5 = vle16_v_f16m2(i5, vl); i5 += vl;
//                 vfloat16m2_t vi6 = vle16_v_f16m2(i6, vl); i6 += vl;
//                 vfloat16m2_t vi7 = vle16_v_f16m2(i7, vl); i7 += vl;
//                 vfloat16m2_t vi8 = vle16_v_f16m2(o, vl);

//                 vfloat16m2_t vmax01 = vfmax_vv_f16m2(vi0, vi1, vl);
//                 vfloat16m2_t vmax23 = vfmax_vv_f16m2(vi2, vi3, vl);
//                 vfloat16m2_t vmax45 = vfmax_vv_f16m2(vi4, vi5, vl);
//                 vfloat16m2_t vmax67 = vfmax_vv_f16m2(vi6, vi7, vl);
//                 vfloat16m2_t vmax018 = vfmax_vv_f16m2(vmax01, vi8, vl);

//                 vfloat16m2_t vmax2345 = vfmax_vv_f16m2(vmax23, vmax45, vl);
//                 vfloat16m2_t vmax01678 = vfmax_vv_f16m2(vmax018, vmax67, vl);
//                 vfloat16m2_t vout = vfmax_vv_f16m2(vmax2345, vmax01678, vl);
//                 vout = vfmax_vf_f16m2(vout, voutput_min, vl);
//                 vout = vfmin_vf_f16m2(vout, voutput_max, vl);

//                 vse16_v_f16m2(o, vout, vl);

//                 o += vl;
//                 c -= vl;
//             } while (c != 0);
//         }
//         input = (const void**) ((uintptr_t) input + input_increment);
//         output = (__fp16*) ((uintptr_t) o + output_increment);
//     } while (--output_pixels != 0);
// }

// void xnn_f16_vsigmoid_ukernel__rvv_u2v(
//         size_t batch,
//         const void* input,
//         void* output,
//         const union xnn_f16_sigmoid_params params[restrict XNN_MIN_ELEMENTS(1)])
// {
//     assert(batch != 0);
//     assert(batch % sizeof(__fp16) == 0);
//     assert(input != NULL);
//     assert(output != NULL);

//     __fp16 vmagic_bias;
//     __fp16 vminus_log2e;
//     uint16_t vindex_mask = UINT16_C(0x3F);
//     __fp16 vln2_hi;
//     __fp16 vln2_lo;
//     __fp16 vc1;
//     __fp16 vc2;
//     __fp16 vone = 1.0f;
//     __fp16 vdenorm_cutoff;

//     memcpy(&vmagic_bias, &params->fp16arith_rr2_p2.magic_bias, sizeof(vmagic_bias));
//     memcpy(&vminus_log2e, &params->fp16arith_rr2_p2.minus_log2e, sizeof(vminus_log2e));

//     memcpy(&vln2_hi, &params->fp16arith_rr2_p2.ln2_hi, sizeof(vln2_hi));
//     memcpy(&vln2_lo, &params->fp16arith_rr2_p2.ln2_lo, sizeof(vln2_lo));
//     memcpy(&vc1, &params->fp16arith_rr2_p2.c1, sizeof(vc1));
//     memcpy(&vc2, &params->fp16arith_rr2_p2.c2, sizeof(vc2));

//     memcpy(&vdenorm_cutoff, &params->fp16arith_rr2_p2.denorm_cutoff, sizeof(vdenorm_cutoff));

//     size_t size = batch / sizeof(__fp16);
//     do {
//         const size_t vl = vsetvl_e16m2(size);
//         vfloat16m2_t vx = vle16_v_f16m2(input, vl);
//         input += vl;
//         // get abs
//         vfloat16m2_t vz = vfabs_v_f16m2(vx, vl);
//         // vz*(-log2(e))+magic_bias
//         vfloat16m2_t vn = vfadd_vf_f16m2(vfmul_vf_f16m2(vz, vminus_log2e, vl), vmagic_bias, vl);
//         // get exponent
//         //vuint16m2_t ve = vsll_vx_u16m2(vreinterpret_v_f16m2_u16m2(vn), 17, vl);
//         // find index in lookup table using mask
//         //vuint16m2_t vidx = vand_vx_u16m2(vreinterpret_v_f16m2_u16m2(vn), vindex_mask, vl);
//         //vfloat16m2_t vs = vreinterpret_v_u16m2_f16m2(vadd_vv_u16m2(vloxei16_v_u16m2(xnn_table_exp2minus_k_over_64, vmul_vx_u16m2(vidx, 4, vl), vl), ve, vl));
//         vfloat16m2_t vs = vreinterpret_v_u16m2_f16m2(vsll_vx_u16m2(vreinterpret_v_f16m2_u16m2(vn), 10, vl));
//         // remove magic bias
//         vn = vfsub_vf_f16m2(vn, vmagic_bias, vl);
//         // find logarithm
//         vfloat16m2_t vt = vfadd_vv_f16m2(vfmul_vf_f16m2(vn, vln2_hi, vl), vz, vl);
//         vt = vfmacc_vf_f16m2(vt, vln2_lo, vn, vl);
//         // calculate the quadratic term logarithmically.
//         //vfloat16m2_t vp = vfmul_vf_f16m2(vt, vc2, vl);
//         //vp = vfsub_vv_f16m2(vt, vfmul_vv_f16m2(vp, vt, vl), vl);
//         vfloat16m2_t vp = vfadd_vf_f16m2(vfmul_vf_f16m2(vt, vc2, vl), vc1, vl);
//         vt = vfmul_vv_f16m2(vt, vs, vl);
//         vfloat16m2_t ve = vfmacc_vv_f16m2(vs, vp, vt, vl);
//         // caculate sigmoid polynomial approximation
//         //vfloat16m2_t vy = vfsub_vv_f16m2(vs, vfmul_vv_f16m2(vs, vp, vl), vl);
//         vfloat16m2_t vd = vfadd_vf_f16m2(ve, vone, vl);
//         vfloat16m2_t vr = vfrdiv_vf_f16m2(vd, 1.0f, vl);
//         //vfloat16m2_t vf = vfdiv_vv_f16m2(vy, vd, vl);

//         vfloat16m2_t vadj = vfadd_vf_f16m2(vfneg_v_f16m2(vfmul_vv_f16m2(vr, vd, vl), vl), 2.0f, vl);

//         vr = vfmul_vv_f16m2(vr, vadj, vl);
//         vfloat16m2_t vf = vfmul_vv_f16m2(ve, vr, vl);

//         vbool8_t mask = vmfgt_vf_f16m2_b8(vz, vdenorm_cutoff, vl);
//         vf = vfmerge_vfm_f16m2(mask, vf, 0.0f, vl);

//         mask = vmfgt_vf_f16m2_b8(vx, 0.0f, vl);
//         vf = vfneg_v_f16m2_m(mask, vf, vf, vl);
//         vf = vfadd_vf_f16m2_m(mask, vf, vf, vone, vl);

//         // store result
//         vse16_v_f16m2(output, vf, vl);

//         output += vl;
//         size -= vl;
//     } while (size > 0);
// }

// void xnn_f16_vsigmoid_ukernel__thead_u2v(
//         size_t batch,
//         const void* input,
//         void* output,
//         const union xnn_f16_sigmoid_params params[restrict XNN_MIN_ELEMENTS(1)])
// {
//     assert(batch != 0);
//     assert(batch % sizeof(__fp16) == 0);
//     assert(input != NULL);
//     assert(output != NULL);

//     __fp16 *input_data = (__fp16 *)input;
//     __fp16 *output_data = (__fp16 *)output;

//     size_t size = batch / sizeof(__fp16);
//     while (size > 0) {
//         size_t vl = vsetvl_e16m2(size);

//         vfloat16m2_t _val = vle16_v_f16m2(input_data, vl);  // val
//         _val = vfmul_vf_f16m2(_val, -1.0f, vl);
//         vfloat16m2_t _output_data = exp_ps_vfloat16m2(_val, vl);
//         _output_data = vfadd_vf_f16m2(_output_data, 1.0f, vl);
//         _output_data = vfrdiv_vf_f16m2(_output_data, 1.0f, vl);
//         vse16_v_f16m2(output_data, _output_data, vl);

//         input_data += vl;
//         output_data += vl;
//         size -= vl;
//     }
// }


// //向量加法
// void xnn_f16_vadd_minmax_ukernel__rvv_u2v(
//         size_t batch,
//         const void* input_a,
//         const void* input_b,
//         void* output,
//         const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
// {
//     assert(batch != 0);
//     assert(batch % sizeof(__fp16) == 0);
//     assert(input_a != NULL);
//     assert(input_b != NULL);
//     assert(output != NULL);

//     __fp16 voutput_min;
//     __fp16 voutput_max;
//     memcpy(&voutput_min, &params->fp16arith.min, sizeof(voutput_min));
//     memcpy(&voutput_max, &params->fp16arith.max, sizeof(voutput_max));

//     size_t size = batch / sizeof(__fp16);
//     do {
//     // 动态设置向量长度
//     const size_t vl = vsetvl_e16m2(size);

//     // 加载输入向量
//     vfloat16m2_t va = vle16_v_f16m2(input_a, vl);
//     vfloat16m2_t vb = vle16_v_f16m2(input_b, vl);
//     input_a += vl;
//     input_b += vl;

//     // 执行向量加法
//     vfloat16m2_t vacc = vfadd_vv_f16m2(va, vb, vl);

//     // 应用最小值约束
//     vacc = vfmax_vf_f16m2(vacc, voutput_min, vl);

//     // 应用最大值约束
//     vacc = vfmin_vf_f16m2(vacc, voutput_max, vl);

//     // 存储结果
//     vse16_v_f16m2(output, vacc, vl);
//     output += vl;
//     size -= vl;
//     } while (size > 0);
// }

// //向量加法
// void xnn_f16_vaddc_minmax_ukernel__rvv_u2v(
//         size_t batch,
//         const void* input_a,
//         const void* input_b,
//         void* output,
//         const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
// {
//     assert(batch != 0);
//     assert(batch % sizeof(__fp16) == 0);
//     assert(input_a != NULL);
//     assert(input_b != NULL);
//     assert(output != NULL);

//     __fp16 voutput_min;
//     __fp16 voutput_max;
//     memcpy(&voutput_min, &params->fp16arith.min, sizeof(voutput_min));
//     memcpy(&voutput_max, &params->fp16arith.max, sizeof(voutput_max));
//     const __fp16 vb = *(__fp16 *)input_b;

//     size_t size = batch / sizeof(__fp16);
//     do {
//         // 动态设置向量长度
//         const size_t vl = vsetvl_e16m2(size);

//         // 加载输入向量
//         vfloat16m2_t va = vle16_v_f16m2(input_a, vl);
//         input_a += vl;

//         // 执行向量加法
//         vfloat16m2_t vacc = vfadd_vf_f16m2(va, vb, vl);

//         // 应用最小值约束
//         vacc = vfmax_vf_f16m2(vacc, voutput_min, vl);

//         // 应用最大值约束
//         vacc = vfmin_vf_f16m2(vacc, voutput_max, vl);

//         // 存储结果
//         vse16_v_f16m2(output, vacc, vl);
//         output += vl;
//         size -= vl;
//     } while (size > 0);
// }

// //向量乘法
// void xnn_f16_vmul_minmax_ukernel__rvv_u2v(
//         size_t batch,
//         const void* input_a,
//         const void* input_b,
//         void* output,
//         const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
// {
//     assert(batch != 0);
//     assert(batch % sizeof(__fp16) == 0);
//     assert(input_a != NULL);
//     assert(input_b != NULL);
//     assert(output != NULL);

//     __fp16 voutput_min;
//     __fp16 voutput_max;
//     memcpy(&voutput_min, &params->fp16arith.min, sizeof(voutput_min));
//     memcpy(&voutput_max, &params->fp16arith.max, sizeof(voutput_max));

//     size_t size = batch / sizeof(__fp16);
//     do {
//         // 动态设置向量长度
//         const size_t vl = vsetvl_e16m2(size);

//         // 加载输入向量
//         vfloat16m2_t va = vle16_v_f16m2(input_a, vl);
//         vfloat16m2_t vb = vle16_v_f16m2(input_b, vl);
//         input_a += vl;
//         input_b += vl;

//         // 执行向量乘法
//         vfloat16m2_t vacc = vfmul_vv_f16m2(va, vb, vl);

//         // 应用最小值约束
//         vacc = vfmax_vf_f16m2(vacc, voutput_min, vl);

//         // 应用最大值约束
//         vacc = vfmin_vf_f16m2(vacc, voutput_max, vl);

//         // 存储结果
//         vse16_v_f16m2(output, vacc, vl);
//         output += vl;
//         size -= vl;
//     } while (size > 0);
// }

// //向量乘法
// void xnn_f16_vmulc_minmax_ukernel__rvv_u2v(
//         size_t batch,
//         const void* input_a,
//         const void* input_b,
//         void* output,
//         const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
// {
//     assert(batch != 0);
//     assert(batch % sizeof(__fp16) == 0);
//     assert(input_a != NULL);
//     assert(input_b != NULL);
//     assert(output != NULL);

//     __fp16 voutput_min;
//     __fp16 voutput_max;
//     memcpy(&voutput_min, &params->fp16arith.min, sizeof(voutput_min));
//     memcpy(&voutput_max, &params->fp16arith.max, sizeof(voutput_max));
//     const __fp16 vb = *(__fp16*)input_b;

//     size_t size = batch / sizeof(__fp16);
//     do {
//         // 动态设置向量长度
//         const size_t vl = vsetvl_e16m2(size);

//         // 加载输入向量
//         vfloat16m2_t va = vle16_v_f16m2(input_a, vl);
//         input_a += vl;

//         // 执行向量乘法
//         vfloat16m2_t vacc = vfmul_vf_f16m2(va, vb, vl);

//         // 应用最小值约束
//         vacc = vfmax_vf_f16m2(vacc, voutput_min, vl);

//         // 应用最大值约束
//         vacc = vfmin_vf_f16m2(vacc, voutput_max, vl);

//         // 存储结果
//         vse16_v_f16m2(output, vacc, vl);
//         output += vl;
//         size -= vl;
//     } while (size > 0);
// }
