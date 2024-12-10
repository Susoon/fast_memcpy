#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

template <typename T>
__global__ void test_assign(
        void *dst,
        const void *src,
        uint32_t size,
        uint64_t *time)
{
    uint64_t start = 0;
    uint64_t end = 0;

    uint32_t len = size / sizeof(T);

    T *srcT = (T *)src;
    T *dstT = (T *)dst;

#if 0
    start = clock64();
    for(int i = 0; i < len; i++){
        dstT[i] = srcT[i];
    }
    end = clock64();
#else
    T prev = 0;
    start = clock64();
    for(int i = 0; i < len; i++){
        srcT[i] -= prev;
        prev = srcT[i];
    }
    end = clock64();
#endif

    *time = end - start;
}

template __global__ void test_assign<uint8_t>(void *dst, const void *src, uint32_t size, uint64_t *time);
template __global__ void test_assign<uint16_t>(void *dst, const void *src, uint32_t size, uint64_t *time);
template __global__ void test_assign<uint32_t>(void *dst, const void *src, uint32_t size, uint64_t *time);
template __global__ void test_assign<uint64_t>(void *dst, const void *src, uint32_t size, uint64_t *time);

#if 0
template <typename T>
__global__ void test_assign(
        void *src,
        uint32_t size,
        uint64_t *time)
{
    uint64_t start = 0;
    uint64_t end = 0;

    uint32_t len = size / sizeof(T);

    T *srcT = (T *)src;

    T prev = 0;
    start = clock64();
    for(int i = 0; i < len; i++){
        srcT[i] -= prev;
        prev = srcT[i];
    }
    end = clock64();

    *time = end - start;
}

template __global__ void test_assign<uint8_t>(void *src, uint32_t size, uint64_t *time);
template __global__ void test_assign<uint16_t>(void *src, uint32_t size, uint64_t *time);
template __global__ void test_assign<uint32_t>(void *src, uint32_t size, uint64_t *time);
template __global__ void test_assign<uint64_t>(void *src, uint32_t size, uint64_t *time);
#endif


