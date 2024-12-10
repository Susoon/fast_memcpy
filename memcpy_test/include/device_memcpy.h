#ifndef __DEVICE_MEMCPY_H__
#define __DEVICE_MEMCPY_H__

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

template <typename T>
__global__ void test_memcpy(
        void *dst,
        const void *src,
        uint32_t size,
        uint64_t *time);

template <typename T>
__global__ void test_memcpyAsync(
        void *dst,
        const void *src,
        uint32_t size,
        uint64_t *time);

#endif
