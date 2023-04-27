# fast_memcpy

## How to use
* Just call **fast_copy** instead of memcpy in \_\_global\_\_ function

## How to compile
* Just add followings in your Makefile
```
LIB_INC := -I$(shell pwd)/lib

fast_copy.o : $(shell pwd)/lib/fast_copy.cu
    @nvcc -g -arch=compute_86 --device-c $^ $(LIB_INC)
```
