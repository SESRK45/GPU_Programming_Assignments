#include <stdio.h>
#include <cuda_runtime.h>

__device__ void my_barrier(int *barrier_counter, int total_blocks) {
    __syncthreads();
    
    if (threadIdx.x == 0) {
        atomicAdd(barrier_counter, 1);
        while (atomicAdd(barrier_counter, 0) < total_blocks) {
            // spin-wait
        }
    }
    
    __syncthreads();
}

__global__ void my_kernel(int *counter, int *output) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    output[tid] = tid;
    
    my_barrier(counter, gridDim.x);
    
    if (tid == 0) {
        printf("Barrier passed! Last thread value is %d\n", output[gridDim.x * blockDim.x - 1]);
    }
}

int main() {
    int blocks = 8;
    int threads = 128;
    int *d_counter;
    int *d_out;
    
    cudaMalloc((void**)&d_counter, sizeof(int));
    cudaMemset(d_counter, 0, sizeof(int));
    
    cudaMalloc((void**)&d_out, blocks * threads * sizeof(int));
    
    my_kernel<<<blocks, threads>>>(d_counter, d_out);
    cudaDeviceSynchronize();
    
    cudaFree(d_counter);
    cudaFree(d_out);
    
    return 0;
}
