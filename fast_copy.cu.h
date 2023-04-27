#ifndef __FAST_COPY_H__
#define __FAST_COPY_H__

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

template <typename T>
__device__ void __fast_copy(
          void *dst,
          const void *src,
          int32_t size);

__device__ void fast_copy(
          void *dst,
          const void *src,
          int32_t size);
#endif
