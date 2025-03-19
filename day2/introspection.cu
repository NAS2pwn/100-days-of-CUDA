#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    printf("Device properties :\n");
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Name : %s\n", prop.name);
    printf("Compute capability : %d.%d\n", prop.major, prop.minor);
    printf("Global memory : %zu bytes\n", prop.totalGlobalMem);
    printf("Shared memory per block : %zu bytes\n", prop.sharedMemPerBlock);
    printf("Registers per block : %d\n", prop.regsPerBlock);
    printf("Warp size : %d\n", prop.warpSize);

    return 0;
}