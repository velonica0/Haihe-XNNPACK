#ifndef RISCV_V_071_FIX_H
#define RISCV_V_071_FIX_H

#include "riscv_vector.h"

#define __riscv_vsetvl_e32m1    vsetvl_e32m1       // 设置向量寄存器每次操作的元素个数
#define __riscv_vsetvl_e32m2    vsetvl_e32m2       // 设置向量寄存器每次操作的元素个数
#define __riscv_vsetvl_e32m4    vsetvl_e32m4       // 设置向量寄存器每次操作的元素个数
#define __riscv_vsetvl_e32m8    vsetvl_e32m8       // 设置向量寄存器每次操作的元素个数
#define __riscv_vle32_v_f32m1   vle32_v_f32m1      // 从数组a中加载vl个元素到向量寄存器va中
#define __riscv_vle32_v_f32m2   vle32_v_f32m2      // 从数组a中加载vl个元素到向量寄存器va中
#define __riscv_vle32_v_f32m4   vle32_v_f32m4      // 从数组a中加载vl个元素到向量寄存器va中
#define __riscv_vle32_v_f32m8   vle32_v_f32m8      // 从数组a中加载vl个元素到向量寄存器va中
#define __riscv_vfadd_vv_f32m1  vfadd_vv_f32m1     // 向量寄存器va和向量寄存器vb中vl个元素对应相加，结果为vc
#define __riscv_vfadd_vv_f32m2  vfadd_vv_f32m2     // 向量寄存器va和向量寄存器vb中vl个元素对应相加，结果为vc
#define __riscv_vfadd_vv_f32m4  vfadd_vv_f32m4     // 向量寄存器va和向量寄存器vb中vl个元素对应相加，结果为vc
#define __riscv_vfadd_vv_f32m8  vfadd_vv_f32m8     // 向量寄存器va和向量寄存器vb中vl个元素对应相加，结果为vc
#define __riscv_vse32_v_f32m1   vse32_v_f32m1      // 将向量寄存器中的vl个元素存到数组c中
#define __riscv_vse32_v_f32m2   vse32_v_f32m2      // 将向量寄存器中的vl个元素存到数组c中
#define __riscv_vse32_v_f32m4   vse32_v_f32m4      // 将向量寄存器中的vl个元素存到数组c中
#define __riscv_vse32_v_f32m8   vse32_v_f32m8      // 将向量寄存器中的vl个元素存到数组c中

#define __riscv_vfadd_vf_f32m1             vfadd_vf_f32m1
#define __riscv_vfadd_vf_f32m2             vfadd_vf_f32m2
#define __riscv_vfadd_vf_f32m4             vfadd_vf_f32m4
#define __riscv_vfadd_vf_f32m8             vfadd_vf_f32m8
#define __riscv_vfaddc_vf_f32m1             vfaddc_vf_f32m1
#define __riscv_vfaddc_vf_f32m2             vfaddc_vf_f32m2
#define __riscv_vfaddc_vf_f32m4             vfaddc_vf_f32m4
#define __riscv_vfaddc_vf_f32m8             vfaddc_vf_f32m8
#define __riscv_vfcvt_f_x_v_f32m1          vfcvt_f_x_v_f32m1
#define __riscv_vfcvt_f_x_v_f32m2          vfcvt_f_x_v_f32m2
#define __riscv_vfcvt_f_x_v_f32m4          vfcvt_f_x_v_f32m4
#define __riscv_vfcvt_f_x_v_f32m8          vfcvt_f_x_v_f32m8
#define __riscv_vfcvt_x_f_v_i32m1          vfcvt_x_f_v_i32m1
#define __riscv_vfcvt_x_f_v_i32m2          vfcvt_x_f_v_i32m2
#define __riscv_vfcvt_x_f_v_i32m4          vfcvt_x_f_v_i32m4
#define __riscv_vfcvt_x_f_v_i32m8          vfcvt_x_f_v_i32m8
#define __riscv_vfcvt_xu_f_v_u32m1         vfcvt_xu_f_v_u32m1
#define __riscv_vfcvt_xu_f_v_u32m2         vfcvt_xu_f_v_u32m2
#define __riscv_vfcvt_xu_f_v_u32m4         vfcvt_xu_f_v_u32m4
#define __riscv_vfcvt_xu_f_v_u32m8         vfcvt_xu_f_v_u32m8
#define __riscv_vfmacc_vf_f32m1            vfmacc_vf_f32m1
#define __riscv_vfmacc_vf_f32m2            vfmacc_vf_f32m2
#define __riscv_vfmacc_vf_f32m4            vfmacc_vf_f32m4
#define __riscv_vfmacc_vf_f32m8            vfmacc_vf_f32m8
#define __riscv_vfmadd_vv_f32m1            vfmadd_vv_f32m1
#define __riscv_vfmadd_vv_f32m2            vfmadd_vv_f32m2
#define __riscv_vfmadd_vv_f32m4            vfmadd_vv_f32m4
#define __riscv_vfmadd_vv_f32m8            vfmadd_vv_f32m8
#define __riscv_vfmax_vf_f32m1             vfmax_vf_f32m1
#define __riscv_vfmax_vf_f32m2             vfmax_vf_f32m2
#define __riscv_vfmax_vf_f32m4             vfmax_vf_f32m4
#define __riscv_vfmax_vf_f32m8             vfmax_vf_f32m8
#define __riscv_vfmin_vf_f32m1             vfmin_vf_f32m1
#define __riscv_vfmin_vf_f32m2             vfmin_vf_f32m2
#define __riscv_vfmin_vf_f32m4             vfmin_vf_f32m4
#define __riscv_vfmin_vf_f32m8             vfmin_vf_f32m8
#define __riscv_vfmul_vf_f32m1             vfmul_vf_f32m1
#define __riscv_vfmul_vf_f32m2             vfmul_vf_f32m2
#define __riscv_vfmul_vf_f32m4             vfmul_vf_f32m4
#define __riscv_vfmul_vf_f32m8             vfmul_vf_f32m8
#define __riscv_vfmul_vv_f32m1             vfmul_vv_f32m1
#define __riscv_vfmul_vv_f32m2             vfmul_vv_f32m2
#define __riscv_vfmul_vv_f32m4             vfmul_vv_f32m4
#define __riscv_vfmul_vv_f32m8             vfmul_vv_f32m8
#define __riscv_vfmv_f_s_f32m1_f32         vfmv_f_s_f32m1_f32
#define __riscv_vfmv_f_s_f32m2_f32         vfmv_f_s_f32m2_f32
#define __riscv_vfmv_f_s_f32m4_f32         vfmv_f_s_f32m4_f32
#define __riscv_vfmv_f_s_f32m8_f32         vfmv_f_s_f32m8_f32
#define __riscv_vfmv_v_f_f32m1             vfmv_v_f_f32m1
#define __riscv_vfmv_v_f_f32m2             vfmv_v_f_f32m2
#define __riscv_vfmv_v_f_f32m4             vfmv_v_f_f32m4
#define __riscv_vfmv_v_f_f32m8             vfmv_v_f_f32m8
#define __riscv_vfncvt_x_f_w_i16m1         vfncvt_x_f_w_i16m1
#define __riscv_vfncvt_x_f_w_i16m2         vfncvt_x_f_w_i16m2
#define __riscv_vfncvt_x_f_w_i16m4         vfncvt_x_f_w_i16m4
#define __riscv_vfncvt_x_f_w_i16m8         vfncvt_x_f_w_i16m8
#define __riscv_vfnmsac_vf_f32m1           vfnmsac_vf_f32m1
#define __riscv_vfnmsac_vf_f32m2           vfnmsac_vf_f32m2
#define __riscv_vfnmsac_vf_f32m4           vfnmsac_vf_f32m4
#define __riscv_vfnmsac_vf_f32m8           vfnmsac_vf_f32m8
#define __riscv_vfsub_vf_f32m1             vfsub_vf_f32m1
#define __riscv_vfsub_vf_f32m2             vfsub_vf_f32m2
#define __riscv_vfsub_vf_f32m4             vfsub_vf_f32m4
#define __riscv_vfsubc_vf_f32m8             vfsubc_vf_f32m8
#define __riscv_vfsubc_vf_f32m1             vfsubc_vf_f32m1
#define __riscv_vfsubc_vf_f32m2             vfsubc_vf_f32m2
#define __riscv_vfsubc_vf_f32m4             vfsubc_vf_f32m4
#define __riscv_vfsub_vf_f32m8             vfsub_vf_f32m8
#define __riscv_vfrsub_vf_f32m1             vfrsub_vf_f32m1
#define __riscv_vfrsub_vf_f32m2             vfrsub_vf_f32m2
#define __riscv_vfrsub_vf_f32m4             vfrsub_vf_f32m4
#define __riscv_vfrsub_vf_f32m8             vfrsub_vf_f32m8
#define __riscv_vfwcvt_f_x_v_f32m1         vfwcvt_f_x_v_f32m1
#define __riscv_vfwcvt_f_x_v_f32m2         vfwcvt_f_x_v_f32m2
#define __riscv_vfwcvt_f_x_v_f32m4         vfwcvt_f_x_v_f32m4
#define __riscv_vfwcvt_f_x_v_f32m8         vfwcvt_f_x_v_f32m8
#define __riscv_vle32_v_f32m1              vle32_v_f32m1
#define __riscv_vle32_v_f32m2              vle32_v_f32m2
#define __riscv_vle32_v_f32m4              vle32_v_f32m4
#define __riscv_vle32_v_f32m8              vle32_v_f32m8
#define __riscv_vle8_v_i8m1                vle8_v_i8m1
#define __riscv_vle8_v_i8m2                vle8_v_i8m2
#define __riscv_vle8_v_i8m4                vle8_v_i8m4
#define __riscv_vle8_v_i8m8                vle8_v_i8m8
#define __riscv_vle8_v_u8m1                vle8_v_u8m1
#define __riscv_vle8_v_u8m2                vle8_v_u8m2
#define __riscv_vle8_v_u8m4                vle8_v_u8m4
#define __riscv_vle8_v_u8m8                vle8_v_u8m8
#define __riscv_vmv_v_v_f32m1              vmv_v_v_f32m1
#define __riscv_vmv_v_v_f32m2              vmv_v_v_f32m2
#define __riscv_vmv_v_v_f32m4              vmv_v_v_f32m4
#define __riscv_vmv_v_v_f32m8              vmv_v_v_f32m8
#define __riscv_vncvt_x_x_w_i16m1          vncvt_x_x_w_i16m1
#define __riscv_vncvt_x_x_w_i16m2          vncvt_x_x_w_i16m2
#define __riscv_vncvt_x_x_w_i16m4          vncvt_x_x_w_i16m4
#define __riscv_vncvt_x_x_w_i16m8          vncvt_x_x_w_i16m8
#define __riscv_vncvt_x_x_w_i8m1           vncvt_x_x_w_i8m1
#define __riscv_vncvt_x_x_w_i8m2           vncvt_x_x_w_i8m2
#define __riscv_vncvt_x_x_w_i8m4           vncvt_x_x_w_i8m4
#define __riscv_vncvt_x_x_w_i8m8           vncvt_x_x_w_i8m8
#define __riscv_vncvt_x_x_w_u16m1          vncvt_x_x_w_u16m1
#define __riscv_vncvt_x_x_w_u16m2          vncvt_x_x_w_u16m2
#define __riscv_vncvt_x_x_w_u16m4          vncvt_x_x_w_u16m4
#define __riscv_vncvt_x_x_w_u16m8          vncvt_x_x_w_u16m8
#define __riscv_vncvt_x_x_w_u8m1           vncvt_x_x_w_u8m1
#define __riscv_vncvt_x_x_w_u8m2           vncvt_x_x_w_u8m2
#define __riscv_vncvt_x_x_w_u8m4           vncvt_x_x_w_u8m4
#define __riscv_vncvt_x_x_w_u8m8           vncvt_x_x_w_u8m8
#define __riscv_vreinterpret_v_i32m1_f32m1 vreinterpret_v_i32m1_f32m1
#define __riscv_vreinterpret_v_i32m2_f32m2 vreinterpret_v_i32m2_f32m2
#define __riscv_vreinterpret_v_i32m4_f32m4 vreinterpret_v_i32m4_f32m4
#define __riscv_vreinterpret_v_i32m8_f32m8 vreinterpret_v_i32m8_f32m8
#define __riscv_vreinterpret_v_u16m1_i16m1 vreinterpret_v_u16m1_i16m1
#define __riscv_vreinterpret_v_u16m2_i16m2 vreinterpret_v_u16m2_i16m2
#define __riscv_vreinterpret_v_u16m4_i16m4 vreinterpret_v_u16m4_i16m4
#define __riscv_vreinterpret_v_u16m8_i16m8 vreinterpret_v_u16m8_i16m8
#define __riscv_vse32_v_f32m1              vse32_v_f32m1
#define __riscv_vse32_v_f32m2              vse32_v_f32m2
#define __riscv_vse32_v_f32m4              vse32_v_f32m4
#define __riscv_vse32_v_f32m8              vse32_v_f32m8
#define __riscv_vse8_v_i8m1                vse8_v_i8m1
#define __riscv_vse8_v_i8m2                vse8_v_i8m2
#define __riscv_vse8_v_i8m4                vse8_v_i8m4
#define __riscv_vse8_v_i8m8                vse8_v_i8m8
#define __riscv_vse8_v_u8m1                vse8_v_u8m1
#define __riscv_vse8_v_u8m2                vse8_v_u8m2
#define __riscv_vse8_v_u8m4                vse8_v_u8m4
#define __riscv_vse8_v_u8m8                vse8_v_u8m8
#define __riscv_vsetvl_e32m1               vsetvl_e32m1
#define __riscv_vsetvl_e32m2               vsetvl_e32m2
#define __riscv_vsetvl_e32m4               vsetvl_e32m4
#define __riscv_vsetvl_e32m8               vsetvl_e32m8
#define __riscv_vsetvl_e8m1                vsetvl_e8m1
#define __riscv_vsetvl_e8m2                vsetvl_e8m2
#define __riscv_vsetvl_e8m4                vsetvl_e8m4
#define __riscv_vsetvl_e8m8                vsetvl_e8m8
#define __riscv_vsll_vx_i32m1              vsll_vx_i32m1
#define __riscv_vsll_vx_i32m2              vsll_vx_i32m2
#define __riscv_vsll_vx_i32m4              vsll_vx_i32m4
#define __riscv_vsll_vx_i32m8              vsll_vx_i32m8
#define __riscv_vsub_vx_i32m1              vsub_vx_i32m1
#define __riscv_vsub_vx_i32m2              vsub_vx_i32m2
#define __riscv_vsub_vx_i32m4              vsub_vx_i32m4
#define __riscv_vsub_vx_i32m8              vsub_vx_i32m8
#define __riscv_vsub_vx_u32m1              vsub_vx_u32m1
#define __riscv_vsub_vx_u32m2              vsub_vx_u32m2
#define __riscv_vsub_vx_u32m4              vsub_vx_u32m4
#define __riscv_vsub_vx_u32m8              vsub_vx_u32m8
#define __riscv_vfdiv_vf_f32m1			   vfdiv_vf_f32m1
#define __riscv_vfdiv_vf_f32m2			   vfdiv_vf_f32m2
#define __riscv_vfdiv_vf_f32m4			   vfdiv_vf_f32m4
#define __riscv_vfdiv_vf_f32m8			   vfdiv_vf_f32m8
#define __riscv_vfdivc_vf_f32m1			   vfdivc_vf_f32m1
#define __riscv_vfdivc_vf_f32m2			   vfdivc_vf_f32m2
#define __riscv_vfdivc_vf_f32m4			   vfdivc_vf_f32m4
#define __riscv_vfdivc_vf_f32m8			   vfdivc_vf_f32m8
#define __riscv_vfrdiv_vf_f32m1			   vfrdiv_vf_f32m1
#define __riscv_vfrdiv_vf_f32m2			   vfrdiv_vf_f32m2
#define __riscv_vfrdiv_vf_f32m4			   vfrdiv_vf_f32m4
#define __riscv_vfrdiv_vf_f32m8			   vfrdiv_vf_f32m8
#define __riscv_vwadd_vx_i32m1             vwadd_vx_i32m1
#define __riscv_vwadd_vx_i32m2             vwadd_vx_i32m2
#define __riscv_vwadd_vx_i32m4             vwadd_vx_i32m4
#define __riscv_vwadd_vx_i32m8             vwadd_vx_i32m8
#define __riscv_vwmul_vv_i32m1             vwmul_vv_i32m1
#define __riscv_vwmul_vv_i32m2             vwmul_vv_i32m2
#define __riscv_vwmul_vv_i32m4             vwmul_vv_i32m4
#define __riscv_vwmul_vv_i32m8             vwmul_vv_i32m8
#define __riscv_vwmul_vx_i32m1             vwmul_vx_i32m1
#define __riscv_vwmul_vx_i32m2             vwmul_vx_i32m2
#define __riscv_vwmul_vx_i32m4             vwmul_vx_i32m4
#define __riscv_vwmul_vx_i32m8             vwmul_vx_i32m8
#define __riscv_vwsubu_vx_u16m1            vwsubu_vx_u16m1
#define __riscv_vwsubu_vx_u16m2            vwsubu_vx_u16m2
#define __riscv_vwsubu_vx_u16m4            vwsubu_vx_u16m4
#define __riscv_vwsubu_vx_u16m8            vwsubu_vx_u16m8
#define __riscv_vwsub_vx_i16m1             vwsub_vx_i16m1
#define __riscv_vwsub_vx_i16m2             vwsub_vx_i16m2
#define __riscv_vwsub_vx_i16m4             vwsub_vx_i16m4
#define __riscv_vwsub_vx_i16m8             vwsub_vx_i16m8
#define __riscv_vfsqrt_v_f32m1			   vfsqrt_v_f32m1
#define __riscv_vfsqrt_v_f32m2			   vfsqrt_v_f32m2
#define __riscv_vfsqrt_v_f32m4			   vfsqrt_v_f32m4
#define __riscv_vfsqrt_v_f32m8			   vfsqrt_v_f32m8
#define __riscv_vfabs_v_f32m1			   vfabs_v_f32m1
#define __riscv_vfabs_v_f32m2			   vfabs_v_f32m2
#define __riscv_vfabs_v_f32m4			   vfabs_v_f32m4
#define __riscv_vfabs_v_f32m8			   vfabs_v_f32m8
#define __riscv_vfneg_v_f32m1			   vfneg_v_f32m1
#define __riscv_vfneg_v_f32m2			   vfneg_v_f32m2
#define __riscv_vfneg_v_f32m4			   vfneg_v_f32m4
#define __riscv_vfneg_v_f32m8			   vfneg_v_f32m8

#define __riscv_vfadd_vv_f32m1_tu(vsum, vsum1, vexp, vl)	vfadd_vv_f32m1(vsum, vexp, vl)
#define __riscv_vfadd_vv_f32m2_tu(vsum, vsum1, vexp, vl)	vfadd_vv_f32m2(vsum, vexp, vl)
#define __riscv_vfadd_vv_f32m4_tu(vsum, vsum1, vexp, vl)	vfadd_vv_f32m4(vsum, vexp, vl)
#define __riscv_vfadd_vv_f32m8_tu(vsum, vsum1, vexp, vl)	vfadd_vv_f32m8(vsum, vexp, vl)
#define __riscv_vfmax_vv_f32m1_tu(t0, t1, vec, vl)		vfmax_vv_f32m1(t0, vec, vl)
#define __riscv_vfmax_vv_f32m2_tu(t0, t1, vec, vl)		vfmax_vv_f32m2(t0, vec, vl)
#define __riscv_vfmax_vv_f32m4_tu(t0, t1, vec, vl)		vfmax_vv_f32m4(t0, vec, vl)
#define __riscv_vfmax_vv_f32m8_tu(t0, t1, vec, vl)		vfmax_vv_f32m8(t0, vec, vl)
#define __riscv_vfmin_vv_f32m1_tu(t0, t1, vec, vl)		vfmin_vv_f32m1(t0, vec, vl)
#define __riscv_vfmin_vv_f32m2_tu(t0, t1, vec, vl)		vfmin_vv_f32m2(t0, vec, vl)
#define __riscv_vfmin_vv_f32m4_tu(t0, t1, vec, vl)		vfmin_vv_f32m4(t0, vec, vl)
#define __riscv_vfmin_vv_f32m8_tu(t0, t1, vec, vl)		vfmin_vv_f32m8(t0, vec, vl)

//#define __riscv_vfredusum_vs_f32m4_f32m1   vfredusum_vs_f32m4_f32m1
//#define __riscv_vfredmax_vs_f32m8_f32m1    vfredmax_vs_f32m8_f32m1
//#define __riscv_vfredmin_vs_f32m8_f32m1    vfredmin_vs_f32m8_f32m1
//#define __riscv_vfmv_s_f_f32m1(src, v1)             vfmv_s_f_f32m1(v1)
#endif 
