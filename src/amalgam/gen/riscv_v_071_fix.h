#ifndef RISCV_V_071_FIX_H
#define RISCV_V_071_FIX_H

#include <riscv_vector.h>

#define __riscv_vsetvl_e32m1    vsetvl_e32m1       // 设置向量寄存器每次操作的元素个数
#define __riscv_vle32_v_f32m1   vle32_v_f32m1      // 从数组a中加载vl个元素到向量寄存器va中
#define __riscv_vfadd_vv_f32m1  vfadd_vv_f32m1     // 向量寄存器va和向量寄存器vb中vl个元素对应相加，结果为vc
#define __riscv_vse32_v_f32m1   vse32_v_f32m1      // 将向量寄存器中的vl个元素存到数组c中

#define __riscv_vfadd_vf_f32m8             vfadd_vf_f32m8
#define __riscv_vfcvt_f_x_v_f32m8          vfcvt_f_x_v_f32m8
#define __riscv_vfcvt_x_f_v_i32m8          vfcvt_x_f_v_i32m8
#define __riscv_vfcvt_xu_f_v_u32m8         vfcvt_xu_f_v_u32m8
#define __riscv_vfmacc_vf_f32m4            vfmacc_vf_f32m4
#define __riscv_vfmadd_vv_f32m4            vfmadd_vv_f32m4

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

#define __riscv_vfmv_f_s_f32m1_f32         vfmv_f_s_f32m1_f32
#define __riscv_vfmv_v_f_f32m4             vfmv_v_f_f32m4
#define __riscv_vfncvt_x_f_w_i16m2         vfncvt_x_f_w_i16m2
#define __riscv_vfnmsac_vf_f32m4           vfnmsac_vf_f32m4
#define __riscv_vfsub_vf_f32m1             vfsub_vf_f32m1
#define __riscv_vfsub_vf_f32m2             vfsub_vf_f32m2
#define __riscv_vfsub_vf_f32m4             vfsub_vf_f32m4
#define __riscv_vfwcvt_f_x_v_f32m4         vfwcvt_f_x_v_f32m4
#define __riscv_vle32_v_f32m2              vle32_v_f32m2
#define __riscv_vle32_v_f32m4              vle32_v_f32m4
#define __riscv_vle32_v_f32m8              vle32_v_f32m8
#define __riscv_vle8_v_i8m2                vle8_v_i8m2
#define __riscv_vle8_v_u8m2                vle8_v_u8m2
#define __riscv_vmv_v_v_f32m8              vmv_v_v_f32m8
#define __riscv_vncvt_x_x_w_i16m4          vncvt_x_x_w_i16m4
#define __riscv_vncvt_x_x_w_i8m2           vncvt_x_x_w_i8m2
#define __riscv_vncvt_x_x_w_u16m4          vncvt_x_x_w_u16m4
#define __riscv_vncvt_x_x_w_u8m2           vncvt_x_x_w_u8m2
#define __riscv_vreinterpret_v_i32m4_f32m4 vreinterpret_v_i32m4_f32m4
#define __riscv_vreinterpret_v_u16m4_i16m4 vreinterpret_v_u16m4_i16m4
#define __riscv_vse32_v_f32m1              vse32_v_f32m1
#define __riscv_vse32_v_f32m2              vse32_v_f32m2
#define __riscv_vse32_v_f32m4              vse32_v_f32m4
#define __riscv_vse32_v_f32m8              vse32_v_f32m8
#define __riscv_vse8_v_i8m2                vse8_v_i8m2
#define __riscv_vse8_v_u8m2                vse8_v_u8m2
#define __riscv_vsetvl_e32m1               vsetvl_e32m1
#define __riscv_vsetvl_e32m2               vsetvl_e32m2
#define __riscv_vsetvl_e32m4               vsetvl_e32m4
#define __riscv_vsetvl_e32m8               vsetvl_e32m8
#define __riscv_vsetvl_e8m2                vsetvl_e8m2
#define __riscv_vsll_vx_i32m4              vsll_vx_i32m4
#define __riscv_vsub_vx_i32m8              vsub_vx_i32m8
#define __riscv_vsub_vx_u32m8              vsub_vx_u32m8
#define __riscv_vwadd_vx_i32m4             vwadd_vx_i32m4
#define __riscv_vwmul_vv_i32m8             vwmul_vv_i32m8
#define __riscv_vwmul_vx_i32m8             vwmul_vx_i32m8
#define __riscv_vwsubu_vx_u16m4            vwsubu_vx_u16m4
#define __riscv_vwsub_vx_i16m4             vwsub_vx_i16m4
// #define __riscv_vmflt_vf_f32m4_b8          vmflt_vf_f32m4_b8 
// #define __riscv_vmflt_vf_f32m8_b4          vmflt_vf_f32m8_b4
// #define __riscv_vfmul_vf_f32m4_m           vfmul_vf_f32m4_m
// #define __riscv_vfmul_vf_f32m8_m           vfmul_vf_f32m8_m

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

// #define __riscv_vfredusum_vs_f32m4_f32m1   vfredusum_vs_f32m4_f32m1
// #define __riscv_vfredmax_vs_f32m8_f32m1    vfredmax_vs_f32m8_f32m1
// #define __riscv_vfredmin_vs_f32m8_f32m1    vfredmin_vs_f32m8_f32m1
// #define __riscv_vfmv_s_f_f32m1(src, v1)             vfmv_s_f_f32m1(v1)

#define __riscv_vfmacc_vf_f32m1             vfmacc_vf_f32m1 
#define __riscv_vfmacc_vf_f32m2             vfmacc_vf_f32m2 
#define __riscv_vlse32_v_f32m1              vlse32_v_f32m1    
#define __riscv_vlse32_v_f32m2              vlse32_v_f32m2    
#define __riscv_vfadd_vf_f32m1              vfadd_vf_f32m1
#define __riscv_vfadd_vf_f32m2              vfadd_vf_f32m2
#define __riscv_vfrdiv_vf_f32m1             vfrdiv_vf_f32m1
#define __riscv_vfrdiv_vf_f32m2             vfrdiv_vf_f32m2
#define __riscv_vfdiv_vv_f32m1              vfdiv_vv_f32m1
#define __riscv_vfdiv_vv_f32m2              vfdiv_vv_f32m2
#define __riscv_vfdiv_vf_f32m1              vfdiv_vf_f32m1
#define __riscv_vfdiv_vf_f32m2              vfdiv_vf_f32m2

#define __riscv_vfmax_vv_f32m1              vfmax_vv_f32m1
#define __riscv_vfmax_vv_f32m2              vfmax_vv_f32m2

#define __riscv_vfmv_v_f_f32m1              vfmv_v_f_f32m1
#define __riscv_vfmv_v_f_f32m2              vfmv_v_f_f32m2   

#define __riscv_vfsub_vv_f32m1            vfsub_vv_f32m1   
#define __riscv_vfsub_vv_f32m2            vfsub_vv_f32m2   


#define __riscv_vsetvl_e16m1                vsetvl_e16m1
#define __riscv_vsetvl_e16m2                vsetvl_e16m2

#define __riscv_vfmacc_vf_f16m1             vfmacc_vf_f16m1   
#define __riscv_vfmacc_vf_f16m2             vfmacc_vf_f16m2
#define __riscv_vfmul_vf_f16m1              vfmul_vf_f16m1
#define __riscv_vfmul_vf_f16m2              vfmul_vf_f16m2
#define __riscv_vfadd_vv_f16m1              vfadd_vv_f16m1
#define __riscv_vfadd_vv_f16m2              vfadd_vv_f16m2
#define __riscv_vfmul_vv_f16m1              vfmul_vv_f16m1
#define __riscv_vfmul_vv_f16m2              vfmul_vv_f16m2
#define __riscv_vfadd_vf_f16m1              vfadd_vf_f16m1
#define __riscv_vfadd_vf_f16m2              vfadd_vf_f16m2

#define __riscv_vse16_v_f16m1               vse16_v_f16m1
#define __riscv_vse16_v_f16m2               vse16_v_f16m2

#define __riscv_vle16_v_f16m1               vle16_v_f16m1
#define __riscv_vle16_v_f16m2               vle16_v_f16m2

#define __riscv_vfmin_vf_f16m1              vfmin_vf_f16m1
#define __riscv_vfmin_vf_f16m2              vfmin_vf_f16m2
#define __riscv_vfmax_vf_f16m1              vfmax_vf_f16m1
#define __riscv_vfmax_vf_f16m2              vfmax_vf_f16m2

#define __riscv_vfmax_vv_f16m1              vfmax_vv_f16m1  
#define __riscv_vfmax_vv_f16m2              vfmax_vv_f16m2  

#define __riscv_vmflt_vf_f32m8_b4(in_u8v, zero, n) vmflt_vf_f32m8_b4(in_u8v, zero, n)
#define __riscv_vmflt_vf_f32m4_b8(in_u8v, zero, n) vmflt_vf_f32m4_b8(in_u8v, zero, n)
#define  __riscv_vfmul_vf_f32m8_m(mask, in_u8v, vslope, n) vfmul_vf_f32m8_m(mask, in_u8v, in_u8v, vslope, n)
#define  __riscv_vfmul_vf_f32m4_m(mask, in_u8v, vslope, n) vfmul_vf_f32m4_m(mask, in_u8v, in_u8v, vslope, n)

#define __riscv_vfrsub_vf_f32m1          vfrsub_vf_f32m1  
#define __riscv_vfrsub_vf_f32m2          vfrsub_vf_f32m2 

#define __riscv_vfrdiv_vf_f16m1          vfrdiv_vf_f16m1
#define __riscv_vfrdiv_vf_f16m2          vfrdiv_vf_f16m2

#define  __riscv_vfmv_s_f_f32m1(INFINITY, 1)  vfmv_s_f_f32m1(INFINITY, 1)


#endif 
