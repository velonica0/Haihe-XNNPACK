// Copyright 2024 Google LLC
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

#include <xnnpack/vbinary.h>

