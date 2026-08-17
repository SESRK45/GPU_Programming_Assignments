#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int dev_count = 0;
    cudaError_t err = cudaGetDeviceCount(&dev_count);

    if (err != cudaSuccess) {
        printf("Error getting device count: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (dev_count == 0) {
        printf("No CUDA capable devices found.\n");
        return 0;
    }

    printf("Found %d CUDA device(s).\n\n", dev_count);

    for (int i = 0; i < dev_count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        printf("--- Device %d: %s ---\n", i, prop.name);
        printf("Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("Total global memory: %lu bytes\n", (unsigned long)prop.totalGlobalMem);
        printf("Shared memory per block: %lu bytes\n", (unsigned long)prop.sharedMemPerBlock);
        printf("Registers per block: %d\n", prop.regsPerBlock);
        printf("Warp size: %d\n", prop.warpSize);
        printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("Max threads dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("Clock rate: %d kHz\n", prop.clockRate);
        printf("Total constant memory: %lu bytes\n", (unsigned long)prop.totalConstMem);
        printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
        printf("L2 cache size: %d bytes\n", prop.l2CacheSize);
        printf("Memory bus width: %d bits\n", prop.memoryBusWidth);
        printf("Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
        printf("\n");
    }

    return 0;
}
