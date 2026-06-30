#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>

// tries to allocate covariance matrix C (n x n floats) and reports result
bool test(const int size) {
    const size_t n = size * size;
    const size_t bytes = n * n * sizeof(float);
    const double gigabytes = static_cast<double>(bytes) / 1000000000.0;

    std::cout << std::fixed << std::setprecision(2)
        << "C " << std::setw(3) << size << "x" << std::left << std::setw(3) << size
        << " n=" << std::setw(6) << n << " -> " << gigabytes << " GB ... ";

    float *d_C;
    const cudaError_t e = cudaMalloc(&d_C, bytes);
    const bool success = e == cudaSuccess;
    const char *message = success ? "OK" : cudaGetErrorString(e);

    std::cout << message << std::endl;

    if (success) {
        cudaFree(d_C);
    }

    return success;
}

int main() {
    size_t free_memory, total;
    cudaMemGetInfo(&free_memory, &total);
    const double free_gb = static_cast<double>(free_memory) / 1000000000.0;
    const double total_gb = static_cast<double>(total) / 1000000000.0;

    std::cout << std::fixed << std::setprecision(2)
        << "VRAM free: " << free_gb << " GB / total: " << total_gb << " GB\n"
        << std::endl;

    int low = 64;

    if (!test(low)) {
        std::cout << "Cannot fit even size=64" << std::endl;
        return 1;
    }

    int high = 512;
    int best = 64;
    for (int step = 0; step < 10; step++) {
        if (low >= high) {
            break;
        }

        const int mid = low + (high - low + 1) / 2;
        if (test(mid)) {
            best = mid;
            low = mid;
        } else {
            high = mid - 1;
        }
    }

    std::cout << "\nMax fit: " << best << "x" << best << std::endl;

    return 0;
}
