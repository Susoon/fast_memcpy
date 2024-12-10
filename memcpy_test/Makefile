INC_DIR := $(shell pwd)/include
SRC_DIR := $(shell pwd)/src
INCLUDE := -I$(INC_DIR)

WARNING := -w

.PHONY: all clean

all: memcpy_test

memcpy_test: assembly.o assign.o device_memcpy.o link.o memcpy_test.o 
	@nvcc -o $@ -arch=compute_86 $^ $(INCLUDE) -L/usr/local/cuda/lib64 -lcudart -lcuda $(WARNING)

memcpy_test.o: $(SRC_DIR)/memcpy_test.cu
	@nvcc -arch=compute_86 --device-c $^ $(INCLUDE) $(WARNING)
	
assembly.o: $(SRC_DIR)/assembly.cu
	@nvcc -arch=compute_86 --device-c $^ $(INCLUDE) $(WARNING) 
	
assign.o: $(SRC_DIR)/assign.cu
	@nvcc -arch=compute_86 --device-c $^ $(INCLUDE) $(WARNING) 
	
device_memcpy.o: $(SRC_DIR)/device_memcpy.cu
	@nvcc -arch=compute_86 --device-c $^ $(INCLUDE) $(WARNING) 

link.o: memcpy_test.o assembly.o assign.o device_memcpy.o
	@nvcc -arch=compute_86 --device-link $^ $(INCLUDE) --output-file $@ $(WARNING)

clean:
	@rm -rf memcpy_test
	@rm -rf *.o
