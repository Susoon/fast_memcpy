#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string>
#include <string.h>
#include <getopt.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>

#include "assembly.h"
#include "assign.h"
#include "device_memcpy.h"

using namespace std;

void getSize(uint32_t size, char *size_str, uint32_t pos)
{
    double tmp_size = size;

    if(size >= 1024 * 1024 * 1024){  // Larger than 1GB
        tmp_size /= 1024 * 1024 * 1024;
        string num_str = to_string(tmp_size);
        string num_strCut = num_str.substr(0, num_str.find('.') + pos + 1);

        memcpy(size_str, num_strCut.c_str(), num_strCut.length());

        size_str[num_strCut.length()] = 'G';
    }
    else if(size >= 1024 * 1024){  // Larger than 1MB
        tmp_size /= 1024 * 1024;
        string num_str = to_string(tmp_size);
        string num_strCut = num_str.substr(0, num_str.find('.') + pos + 1);

        memcpy(size_str, num_strCut.c_str(), num_strCut.length());

        size_str[num_strCut.length()] = 'M';
    }
    else if(size >= 1024){  // Larger than 1KB
        tmp_size /= 1024;
        string num_str = to_string(tmp_size);
        string num_strCut = num_str.substr(0, num_str.find('.') + pos + 1);

        memcpy(size_str, num_strCut.c_str(), num_strCut.length());

        size_str[num_strCut.length()] = 'K';
    }
    else{
        string num_str = to_string((int)tmp_size);

        memcpy(size_str, num_str.c_str(), num_str.length());
    }
}

int getSize(char *size_str)
{
    int size = 0;

    int len = strlen(size_str);

    if(size_str[len - 1] == 'K'){
        size_str[len - 1] = 0;
        string num_str(size_str);
        size = stoi(num_str);
        size *= 1024;
    }
    else if(size_str[len - 1] == 'M'){
        size_str[len - 1] = 0;
        string num_str(size_str);
        size = stoi(num_str);
        size *= 1024 * 1024;
    }
    else if(size_str[len - 1] == 'G'){
        size_str[len - 1] = 0;
        string num_str(size_str);
        size = stoi(num_str);
        size *= 1024 * 1024 * 1024;
    }
    else{
        string num_str(size_str);
        size = stoi(num_str);
    }

    return size;
}

string getTime(double time)
{
    int unit_count = 0;
    char unit[] = {'n', 'u', 'm'};
    while(unit_count < 3 && time >= 1){
        time /= 1000;
        unit_count++;
    }
    if(time < 1){
        time *= 1000;
        unit_count--;
    }

    string time_str = to_string(time);
    string time_strCut = time_str.substr(0, time_str.find('.') + 3);

    if(unit_count < 3){
        time_strCut.append(1, unit[unit_count]);
    }
    time_strCut.append(1, 's');
    
    return time_strCut;
}

__device__ int getRand(curandState *s, int A, int B)
{
    float rand_int = curand_uniform(s);
    rand_int = rand_int * (B-A) + A;

    return rand_int;
}

__global__ void set_rand_array(uint8_t *arr, uint32_t len)
{

    curandState s;
    curand_init(1, 0, 0, &s);

    for(int i = 0; i < len - 1; i++){
        arr[i] = getRand(&s, 0, 255);
    }

    arr[len - 1] = -1;
}

    template <typename T>
double run_test_kernel(T test_kernel, uint32_t size, uint32_t clockRate)
{
    uint8_t *in;
    uint8_t *out;

    uint64_t * d_time;
    uint64_t h_time = 0;

    cudaMalloc((void **)&in, size); 
    cudaMalloc((void **)&out, size); 

    set_rand_array <<< 1, 1 >>> (in, size);

    cudaMemset((void *)out, 0, size);

    cudaMalloc((void **)&d_time, sizeof(uint64_t));
    cudaMemset((void *)d_time, 0, sizeof(uint64_t));

    cudaDeviceSynchronize();

    test_kernel <<< 1, 1 >>> (out, in, size, d_time);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_time, d_time, sizeof(uint64_t), cudaMemcpyDeviceToHost);

    cudaFree(in);
    cudaFree(out);
    cudaFree(d_time);

    return (((double)h_time) / clockRate);
}

int main(int argc, char **argv) {
    char *size_str = NULL;
    char asm_test = 'n';
    char assign_test = 'n';
    char memcpy_test = 'n';

    double assign_times[10] = {0.0};
    double assembly_times[10] = {0.0};
    double memcpy_times[2] = {0.0};
    cudaDeviceProp prop;

    uint32_t size = 1000000;

    struct option long_options[] = {
        {"size", required_argument, 0, 's'},
        {"asm", required_argument, 0, 'a'},
        {"ass", required_argument, 0, 'A'},
        {"mcp", required_argument, 0, 'm'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0} // 끝을 알림
    };

    int opt;
    int option_index = 0;

    while ((opt = getopt_long(argc, argv, "s:a:A:m:h", long_options, &option_index)) != -1) {
        switch (opt) {
            case 's': // --size=n
                size_str = optarg;
                size = getSize(size_str);
                break;
            case 'a': // --asm=y/n
                asm_test = optarg[0];
                break;
            case 'A': // --ass=y/n
                assign_test = optarg[0];
                break;
            case 'm': // --mcp=y/n
                memcpy_test = optarg[0];
                break;
            case 'h': // --help
                printf("Usage:\n");
                printf("      --size=n (Unit can be entered like 10M)\n");
                printf("      --asm=y/n (Assembly Test)\n");
                printf("      --ass=y/n (Assign Test)\n");
                printf("      --mcp=y/n (Memcpy Test)\n");
                printf("      --help    (Show this help message)\n");
                exit(EXIT_SUCCESS);
                break;
            default:
                printf("Invalid option. Use --help for usage information.\n");
                exit(EXIT_FAILURE);
        }
    }
    

    cudaGetDeviceProperties(&prop, 0);

    printf("GPU properties!!\n", __FUNCTION__);
    printf("GPU name : %s\n", prop.name);
    printf("clockRate : %d\n", prop.clockRate);
    printf("Memory Bus width : %d\n", prop.memoryBusWidth);
    printf("Memory Bus ClockRate : %d\n", prop.memoryClockRate);


    char size_expression[50] = {0};
    getSize(size, size_expression, 2);

    printf("\nSize of memory : %sB\n", size_expression);

    printf("\nEnabled Test:\n");
    if(asm_test == 'y'){
        printf("Assembly Test\n");
        assembly_times[0] = run_test_kernel(test_asm<uint8_t>, size, prop.clockRate);
        assembly_times[1] = run_test_kernel(test_asm<uint16_t>, size, prop.clockRate);
        assembly_times[2] = run_test_kernel(test_asm<uint32_t>, size, prop.clockRate);
        assembly_times[3] = run_test_kernel(test_asm<uint64_t>, size, prop.clockRate);
    }

    if(assign_test == 'y'){
        printf("Assign Test\n");
        assign_times[0] = run_test_kernel(test_assign<uint8_t>, size, prop.clockRate);
        assign_times[1] = run_test_kernel(test_assign<uint16_t>, size, prop.clockRate);
        assign_times[2] = run_test_kernel(test_assign<uint32_t>, size, prop.clockRate);
        assign_times[3] = run_test_kernel(test_assign<uint64_t>, size, prop.clockRate);
    }

    if(memcpy_test == 'y'){
        printf("Memcpy Test\n");
        memcpy_times[0] = run_test_kernel(test_memcpy<uint8_t>, size, prop.clockRate);
        memcpy_times[1] = run_test_kernel(test_memcpyAsync<uint8_t>, size, prop.clockRate);
    }

    printf("\nElapsed times\n", __FUNCTION__);

    if(asm_test == 'y'){
        printf("\nassembly\n");
        printf(" 8bit  : %s\n", getTime(assembly_times[0]).c_str());
        printf("16bit  : %s\n", getTime(assembly_times[1]).c_str());
        printf("32bit  : %s\n", getTime(assembly_times[2]).c_str());
        printf("64bit  : %s\n", getTime(assembly_times[3]).c_str());
    }

    if(assign_test == 'y'){
        printf("\nassign\n");
        printf(" 8bit  : %s\n", getTime(assign_times[0]).c_str());
        printf("16bit  : %s\n", getTime(assign_times[1]).c_str());
        printf("32bit  : %s\n", getTime(assign_times[2]).c_str());
        printf("64bit  : %s\n", getTime(assign_times[3]).c_str());
    }

    if(memcpy_test == 'y'){
        printf("\nmemcpy\n");
        printf("normal : %s\n", getTime(memcpy_times[0]).c_str());
        printf("Async  : %s\n", getTime(memcpy_times[1]).c_str());
    }

    return 0;
} 
