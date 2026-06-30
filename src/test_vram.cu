#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>

bool test(const int size, const size_t max_allowed) {
    const size_t n = size * size;
    const size_t bytes = n * n * sizeof(float);
    const double gigabytes = static_cast<double>(bytes) / 1000000000.0;
    const bool success = bytes <= max_allowed;
    const char *message = success ? "OK" : "out of memory";

    std::cout << std::fixed << std::setprecision(2)
        << "C " << std::setw(3) << size << "x" << std::left << std::setw(3) << size
        << " n=" << std::setw(6) << n << " -> " << std::right << std::setw(5) << gigabytes << " GB ... "
        << message
        << std::endl;

    return success;
}

int main() {
    size_t free_memory, total;
    cudaMemGetInfo(&free_memory, &total);
    const size_t max_allowed = free_memory - free_memory / 10;
    const double free_gb = static_cast<double>(free_memory) / 1000000000.0;
    const double total_gb = static_cast<double>(total) / 1000000000.0;
    const double max_allowed_gb = static_cast<double>(max_allowed) / 1000000000.0;

    std::cout << std::fixed << std::setprecision(2)
        << "VRAM free: " << free_gb << " GB / total: " << total_gb << " GB\n"
        << "With 10% buffer: " << max_allowed_gb << " GB\n"
        << std::endl;

    int low = 64;

    if (!test(low, max_allowed)) {
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
        if (test(mid, max_allowed)) {
            best = mid;
            low = mid;
        } else {
            high = mid - 1;
        }
    }

    std::cout << "\nMax fit with 10% buffer: " << best << "x" << best << std::endl;

    return 0;
}
