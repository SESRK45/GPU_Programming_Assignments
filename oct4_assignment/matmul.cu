#include <cstdio>
#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>

// Bring the wmma namespace into scope for cleaner code
using namespace nvcuda;
using namespace wmma;

// The Hardware Limit: Tensor Cores process 16x16x16 tiles
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// Problem Size: 64x64 matrices
const int M = 64;
const int N = 64;
const int K = 64;

// Basic initialization kernel (runs on normal scalar ALUs)
__global__ void init_matrices(half *A, half *B, int total_elements) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < total_elements) {
        A[tid] = __float2half((float)(tid % 100));
        B[tid] = __float2half((float)(tid % 100));
    }
}

// Tier-0 WMMA Kernel
__global__ void wmma_gemm(half *A, half *B, float *C, int M, int N, int K) {
    // 1. Identify which 16x16 output tile this WARP is responsible for.
    // We map blockIdx.x to the column (N) and blockIdx.y to the row (M).
    int warp_col = blockIdx.x;
    int warp_row = blockIdx.y;
    
    // Calculate the absolute starting index of this 16x16 tile in the global matrix
    int c_row_start = warp_row * WMMA_M;
    int c_col_start = warp_col * WMMA_N;

    // Bounds check to ensure we don't access outside the matrix
    if (c_row_start >= M || c_col_start >= N) return;

    // 2. Declare the Fragments (Distributed Registers for the 32 threads)
    // matrix_a and matrix_b hold the inputs (half precision)
    // accumulator holds the output (float precision)
    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> frag_A;
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> frag_B;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_C;

    // 3. Initialize the Accumulator Fragment to 0.0f
    fill_fragment(frag_C, 0.0f);

    // 4. Slide across the K dimension in chunks of 16 (WMMA_K)
    // For a 64x64 matrix, this loop runs exactly 4 times (64/16 = 4)
    for (int tile_k = 0; tile_k < K; tile_k += WMMA_K) {
        
        // Calculate memory pointers for this specific 16x16 chunk of A and B
        half *a_tile_ptr = A + (c_row_start * K) + tile_k;
        half *b_tile_ptr = B + (tile_k * N) + c_col_start;

        // Step A: The warp cooperatively loads 256 elements from Global VRAM -> Registers
        // The last argument is the "leading dimension" (stride) of the global matrix
        load_matrix_sync(frag_A, a_tile_ptr, K);
        load_matrix_sync(frag_B, b_tile_ptr, N);
        
        // Step B: Fire the Tensor Core! 
        // Math: frag_C = (frag_A * frag_B) + frag_C
        mma_sync(frag_C, frag_A, frag_B, frag_C);
    }

    // 5. Store the final accumulated 16x16 result back to Global VRAM
    float *c_tile_ptr = C + (c_row_start * N) + c_col_start;
    store_matrix_sync(c_tile_ptr, frag_C, N, mem_row_major);
}

int main() {
    half *d_A, *d_B;
    float *d_C, *h_C;
    cudaEvent_t start, stop;
    float ms = 0.0f;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate memory on Device (GPU) and Host (CPU)
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(float));
    h_C = (float *)malloc(M * N * sizeof(float));

    // Launch normal scalar kernel to initialize A and B
    int total_elements = M * K;
    init_matrices<<<(total_elements + 255) / 256, 256>>>(d_A, d_B, total_elements);
    cudaDeviceSynchronize();

    // Configure the Grid and Block
    // A Warp is 32 threads. We launch 1 Warp per Block here for simplicity.
    // We need 4x4 = 16 Blocks to cover the 64x64 matrix.
    dim3 grid(N / WMMA_N, M / WMMA_M); // Grid size: 4x4
    dim3 block(32);                    // Block size: 32 threads (Exactly 1 Warp)

    // Run and Time the Tensor Core Kernel
    cudaEventRecord(start, 0);
    wmma_gemm<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop, 0);
    
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    
    printf("Tensor Core Math Time: %f ms\n", ms);
    
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // Copy result back to CPU
    cudaMemcpy(h_C, d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    // Print a couple of values to verify
    printf("C[0][0] = %f\n", h_C[0]);
    printf("C[63][63] = %f\n", h_C[63 * N + 63]);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_C);

    return 0;
}
