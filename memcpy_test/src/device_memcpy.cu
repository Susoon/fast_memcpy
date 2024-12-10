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
        uint64_t *time)
{
    uint64_t start = 0;
    uint64_t end = 0;

    start = clock64();
    memcpy(dst, src, size);
    end = clock64();

    *time = end - start;
}

template <typename T>
__global__ void test_memcpyAsync(
        void *dst,
        const void *src,
        uint32_t size,
        uint64_t *time)
{
    uint64_t start = 0;
    uint64_t end = 0;

    cudaError_t err;
    cudaStream_t stream;

    uint8_t *tmp_dst = (uint8_t *)dst;

    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    start = clock64();
    err = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream);
    //while(tmp_dst[size - 1] == 0);
    end = clock64();

    if(err != cudaSuccess)
        printf("[SHKIM:%s] err : %s\n", __FUNCTION__, cudaGetErrorName(err));

    cudaStreamDestroy(stream);

    *time = end - start;
}

template __global__ void test_memcpy<uint8_t>(void *dst, const void *src, uint32_t size, uint64_t *time);
template __global__ void test_memcpyAsync<uint8_t>(void *dst, const void *src, uint32_t size, uint64_t *time);
