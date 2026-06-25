#include <cstdio>
#include <cuda_runtime.h>

// Intenta alojar la matriz de covarianza C (n x n floats) y reporta el resultado.
void probar(size_t n, const char* etiqueta) {
    size_t bytesC = n * n * sizeof(float);
    printf("C %-9s n=%zu -> %.2f GiB ... ", etiqueta, n, bytesC / 1073741824.0);
    float* d_C;
    cudaError_t e = cudaMalloc(&d_C, bytesC);
    printf("%s\n", e == cudaSuccess ? "OK" : cudaGetErrorString(e));
    if (e == cudaSuccess) cudaFree(d_C);
}

int main() {
    size_t libre, total;
    cudaMemGetInfo(&libre, &total);
    printf("VRAM libre: %.2f GiB / total: %.2f GiB\n",
           libre / 1073741824.0, total / 1073741824.0);
    probar(36864, "192x192");
    probar(16384, "128x128");
    return 0;
}
