#ifndef __ASSEMBLY_H__
#define __ASSEMBLY_H__

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

__device__ static inline void copy_u8_with_reg(uint8_t* dst, const uint8_t* src) {
    uint32_t value;
    asm volatile (
            "ld.global.u8 %0, [%1];\n\t"  // Load a byte from src
            "st.global.u8 [%2], %0;\n\t"  // Store the byte to dst
            : "=r"(value)
            : "l"(src), "l"(dst)
            );
}

__device__ static inline void copy_u16_with_reg(uint16_t* dst, const uint16_t* src) {
    uint32_t value;
    asm volatile (
            "ld.global.u16 %0, [%1];\n\t"  // Load a byte from src
            "st.global.u16 [%2], %0;\n\t"  // Store the byte to dst
            : "=r"(value)
            : "l"(src), "l"(dst)
            );
}

__device__ static inline void copy_u32_with_reg(uint32_t* dst, const uint32_t* src) {
    uint32_t value;
    asm volatile (
            "ld.global.u32 %0, [%1];\n\t"  // Load a byte from src
            "st.global.u32 [%2], %0;\n\t"  // Store the byte to dst
            : "=r"(value)
            : "l"(src), "l"(dst)
            );
}

__device__ static inline void copy_u64_with_reg(uint64_t* dst, const uint64_t* src) {
    uint64_t value;
    asm volatile (
            "ld.global.u64 %0, [%1];\n\t"  // Load a byte from src
            "st.global.u64 [%2], %0;\n\t"  // Store the byte to dst
            : "=l"(value)
            : "l"(src), "l"(dst)
            );
}


template <typename T>
__global__ void test_asm(
        void *dst,
        const void *src,
        uint32_t size,
        uint64_t *time);

#if 0
template <typename T>
__global__ void test_asm(
        void *src,
        uint32_t size,
        uint64_t *time);
#endif
#endif
