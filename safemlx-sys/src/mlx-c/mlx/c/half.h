/* Copyright © 2023-2024 Apple Inc. */

#ifndef MLX_HALF_H
#define MLX_HALF_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define HAS_FLOAT16
#if defined(__ARM_FEATURE_FP16_SCALAR_ARITHMETIC) || defined(__aarch64__)
#include <arm_fp16.h>
typedef __fp16 float16_t;
#else
typedef uint16_t float16_t;
#endif

#define HAS_BFLOAT16
#if defined(__ARM_FEATURE_BF16) || defined(__aarch64__)
#include <arm_bf16.h>
typedef __bf16 bfloat16_t;
#else
typedef uint16_t bfloat16_t;
#endif

#ifdef __cplusplus
}
#endif

#endif
