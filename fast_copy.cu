#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "fast_copy.cu.h"

#define ADDR_COPY_OFFSET (8)
#define ADDR_COPY_MASK (ADDR_COPY_OFFSET - 1)
#define COPY64_SIZE_OFFSET (8)
#define COPY64_SIZE_MASK (COPY64_SIZE_OFFSET - 1)

template <typename T>
__device__ void __fast_copy(
        void *dst,
        const void *src,
        int32_t size)
{
    if(size <= 0)
        return;

    uint32_t len = size / sizeof(T);

    T *dstT = (T *)dst;
    const T *srcT = (const T *)src;

    for(uint32_t i = 0; i < len; i++){
        dstT[i] = srcT[i];
    }
}

template __device__ void __fast_copy<uint8_t>(void *dst, const void *src, int32_t size);
template __device__ void __fast_copy<uint16_t>(void *dst, const void *src, int32_t size);
template __device__ void __fast_copy<uint32_t>(void *dst, const void *src, int32_t size);
template __device__ void __fast_copy<uint64_t>(void *dst, const void *src, int32_t size);

__device__ void fast_copy(
        void *dst,
        const void *src,
        int32_t size)
{
    if(((uint64_t)dst & ADDR_COPY_MASK) == ((uint64_t)src & ADDR_COPY_MASK)){
        uint32_t align = (ADDR_COPY_OFFSET - ((uint64_t)dst & ADDR_COPY_MASK)) == 8 ? 0 : (ADDR_COPY_OFFSET - ((uint64_t)dst & ADDR_COPY_MASK));

        uint32_t count = align > size ? size : align;

        __fast_copy<uint8_t>(dst, src, count);

        size -= count;

        if(size <= 0)
            return;

        align = size & (-1 ^ COPY64_SIZE_MASK);

        __fast_copy<uint64_t>((void *)((uint8_t *)dst + count), (void *)((uint8_t *)src + count), align);

        __fast_copy<uint8_t>((void *)((uint8_t *)dst + count + align), (void *)((uint8_t *)src + count + align), size - align);
    }
    else{
        __fast_copy<uint8_t>(dst, src, size);
    }
}

