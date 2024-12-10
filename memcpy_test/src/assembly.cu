#include "assembly.h"

template <typename T>
__global__ void test_asm(
        void *dst,
        const void *src,
        uint32_t size,
        uint64_t *time)
{
    printf("[SHKIM:%s] Wrong Type!!!\n", __FUNCTION__);
/*
    uint64_t start = 0;
    uint64_t end = 0;

    void (*copy)(T*, T*);

    switch(sizeof(T)){
        case sizeof(uint8_t):
            copy = copy_u8_with_reg;
            break;
        case sizeof(uint16_t):
            copy = copy_u16_with_reg;
            break;
        case sizeof(uint32_t):
            copy = copy_u16_with_reg;
            break;
        case sizeof(uint64_t):
            copy = copy_u8_with_reg;
            break;
        default:
            printf("[SHKIM:%s] Wrong Type!!!\n", __FUNCTION__);
            return;
    }

    start = clock64();
    for(int i = 0; i < size / sizeof(T); i++){
        copy((T *)(src + i), (T*)(dst + i));
    }
    end = clock64();

    *time = end - start;
*/
}

template <>
__global__ void test_asm<uint8_t>(
        void *dst,
        const void *src,
        uint32_t size,
        uint64_t *time)
{
    uint64_t start = 0;
    uint64_t end = 0;

    start = clock64();
    for(int i = 0; i < size; i++){
        copy_u8_with_reg((uint8_t *)dst + i, (uint8_t *)src + i);
    }
    end = clock64();

    *time = end - start;
}
        
template <>
__global__ void test_asm<uint16_t>(
        void *dst,
        const void *src,
        uint32_t size,
        uint64_t *time)
{
    uint64_t start = 0;
    uint64_t end = 0;

    uint32_t len = size / sizeof(uint16_t);

    start = clock64();
    for(int i = 0; i < len; i++){
        copy_u16_with_reg((uint16_t *)dst + i, (uint16_t *)src + i);
    }
    end = clock64();

    *time = end - start;
}

template <>
__global__ void test_asm<uint32_t>(
        void *dst,
        const void *src,
        uint32_t size,
        uint64_t *time)
{
    uint64_t start = 0;
    uint64_t end = 0;

    uint32_t len = size / sizeof(uint32_t);

    start = clock64();
    for(int i = 0; i < len; i++){
        copy_u32_with_reg((uint32_t *)dst + i, (uint32_t *)src + i);
    }
    end = clock64();

    *time = end - start;
}

template <>
__global__ void test_asm<uint64_t>(
        void *dst,
        const void *src,
        uint32_t size,
        uint64_t *time)
{
    uint64_t start = 0;
    uint64_t end = 0;

    uint32_t len = size / sizeof(uint64_t);

    start = clock64();
    for(int i = 0; i < len; i++){
        copy_u64_with_reg((uint64_t *)dst + i, (uint64_t *)src + i);
    }
    end = clock64();

    *time = end - start;
}

template <typename T>
__global__ void test_asm(void *dst, const void *src, uint32_t size, uint64_t *time);
template <>
__global__ void test_asm<uint8_t>(void *dst, const void *src, uint32_t size, uint64_t *time);
template <>
__global__ void test_asm<uint16_t>(void *dst, const void *src, uint32_t size, uint64_t *time);
template <>
__global__ void test_asm<uint32_t>(void *dst, const void *src, uint32_t size, uint64_t *time);
template <>
__global__ void test_asm<uint64_t>(void *dst, const void *src, uint32_t size, uint64_t *time);


#if 0
template <typename T>
__global__ void test_asm(
        void *src,
        uint32_t size,
        uint64_t *time)
{
    printf("[SHKIM:%s] Wrong Type!!!\n", __FUNCTION__);
}

template <>
__global__ void test_asm<uint8_t>(
        void *src,
        uint32_t size,
        uint64_t *time)
{
    // Do Nothing
}
        
template <>
__global__ void test_asm<uint16_t>(
        void *src,
        uint32_t size,
        uint64_t *time)
{
    // Do Nothing
}

template <>
__global__ void test_asm<uint32_t>(
        void *src,
        uint32_t size,
        uint64_t *time)
{
    // Do Nothing
}

template <>
__global__ void test_asm<uint64_t>(
        void *src,
        uint32_t size,
        uint64_t *time)
{
    // Do Nothing
}

template <typename T>
__global__ void test_asm(void *src, uint32_t size, uint64_t *time);
template <>
__global__ void test_asm<uint8_t>(void *src, uint32_t size, uint64_t *time);
template <>
__global__ void test_asm<uint16_t>(void *src, uint32_t size, uint64_t *time);
template <>
__global__ void test_asm<uint32_t>(void *src, uint32_t size, uint64_t *time);
template <>
__global__ void test_asm<uint64_t>(void *src, uint32_t size, uint64_t *time);
#endif

