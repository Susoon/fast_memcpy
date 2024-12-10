#ifndef __ASSIGN_H__
#define __ASSIGN_H__

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
          uint64_t *time);

#endif
