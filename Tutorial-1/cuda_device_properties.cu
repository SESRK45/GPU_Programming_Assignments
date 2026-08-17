#include <iostream>
#include <cuda_runtime.h>

using namespace std;

// helper function to get cuda cores per sm
int get_sp_cores(int major, int minor) {
    if (major == 2) return (minor == 1) ? 48 : 32;
    if (major == 3) return 192;
    if (major == 5) return 128;
    if (major == 6) return (minor == 0) ? 64 : 128;
    if (major == 7) return 64;
    if (major == 8) return (minor == 0) ? 64 : 128;
    if (major == 9) return 128;
    return 128;
}

int main() {
    int count = 0;
    cudaGetDeviceCount(&count);

    if (count == 0) {
        cout << "No GPU detected." << endl;
        return 0;
    }

    cout << "Found " << count << " CUDA devices." << endl;

    for (int i = 0; i < count; i++) {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, i);

        int cores = get_sp_cores(p.major, p.minor);
        int total = p.multiProcessorCount * cores;

        cout << "\n--- Device " << i << ": " << p.name << " ---" << endl;
        cout << "Compute capability: " << p.major << "." << p.minor << endl;
        cout << "Streaming Multiprocessors (SMs): " << p.multiProcessorCount << endl;
        cout << "Cores per SM: " << cores << endl;
        cout << "Total CUDA Cores: " << total << endl;
        
        cout << "\nMemory Info:" << endl;
        cout << "Total global mem: " << p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0) << " GB" << endl;
        cout << "Shared mem per block: " << p.sharedMemPerBlock / 1024.0 << " KB" << endl;
        cout << "L2 Cache: " << p.l2CacheSize / (1024.0 * 1024.0) << " MB" << endl;

        cout << "\nThread Limits:" << endl;
        cout << "Max threads per block: " << p.maxThreadsPerBlock << endl;
        cout << "Max threads dims: " << p.maxThreadsDim[0] << "x" << p.maxThreadsDim[1] << "x" << p.maxThreadsDim[2] << endl;
        cout << "Max grid dims: " << p.maxGridSize[0] << "x" << p.maxGridSize[1] << "x" << p.maxGridSize[2] << endl;
        cout << "Warp size: " << p.warpSize << endl;
    }
    
    return 0;
}
