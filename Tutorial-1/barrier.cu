#include <iostream>
#include <cuda_runtime.h>

__device__ void sync_all_blocks(int *counter, int total_blocks) {
    __syncthreads(); // sync threads within block first

    if (threadIdx.x == 0) {
        // block leader increments the counter
        atomicAdd(counter, 1);
        
        // wait until all blocks have incremented
        while (atomicAdd(counter, 0) < total_blocks) {
            // just spin
        }
    }
    __syncthreads(); // sync threads within block again after block leader finishes waiting
}

__global__ void kernel_test_barrier(int *counter, int *out_data) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // write something to memory
    out_data[tid] = tid * 2;
    
    // global barrier call
    sync_all_blocks(counter, gridDim.x);
    
    int last_idx = (gridDim.x * blockDim.x) - 1;

    // test if barrier worked by reading data written by the last thread
    if (tid == 0) {
        printf("Thread 0 reading from last thread (%d): %d\n", last_idx, out_data[last_idx]);
    }
}

int main() {
    int blocks = 4;
    int threads = 256;
    int n = blocks * threads;

    int *d_count;
    int *d_arr;

    cudaMalloc(&d_count, sizeof(int));
    cudaMemset(d_count, 0, sizeof(int));
    
    cudaMalloc(&d_arr, n * sizeof(int));

    std::cout << "Launching kernel with " << blocks << " blocks of " << threads << " threads." << std::endl;
    
    kernel_test_barrier<<<blocks, threads>>>(d_count, d_arr);
    
    cudaDeviceSynchronize();
    
    cudaFree(d_count);
    cudaFree(d_arr);

    return 0;
}
