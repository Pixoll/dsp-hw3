#include "experiment2.hpp"

#include <cstdio>
#include <cuda_runtime.h>

static constexpr int TILE_SIZE = 16;

// Kernel para promedio
__global__ void kernel_add(const float *datos, float *suma_global, const int n) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(&suma_global[idx], datos[idx]);
    }
}

// Lo mismo pero para la covarianza
__global__ void kernel_covariance(const float *batch, float *C_final, const float *mu, const int n) {
    // Memoria compartida: es como una caché ultrarápida dentro del bloque
    __shared__ float s_batch[TILE_SIZE];

    const unsigned int fila = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Solo cargamos el valor una vez en memoria compartida por cada hilo del bloque
    // Esto evita miles de accesos a la memoria global
    if (threadIdx.x < TILE_SIZE && fila < n) {
        s_batch[threadIdx.x] = batch[fila] - mu[fila];
    }
    __syncthreads(); // IMPORTANTE: Esperar a que todos los hilos del bloque carguen el dato

    if (fila < n && col < n) {
        // Ahora usamos los datos de la memoria compartida
        const float val = s_batch[threadIdx.y] * s_batch[threadIdx.x];
        atomicAdd(&C_final[fila * n + col], val);
    }
}

void run_experiment2(const float *h_dataset, const int m, const int n, const int num_streams) {
    // Variables para medir tiempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const size_t n_squared = static_cast<size_t>(n) * n;

    // Reservar memoria en GPU
    float *d_mu, *d_C, *d_batch;
    cudaMalloc(&d_mu, n * sizeof(float)); // Reserva de espacio para el promedio
    cudaMalloc(&d_C, n_squared * sizeof(float)); // Reserva de espacio para la matriz Covarianza
    cudaMalloc(&d_batch, n * sizeof(float)); // Reserva de espacio para 1 imagen

    cudaMemset(d_mu, 0, n * sizeof(float)); // Llenamos de 0 el vector promedio
    cudaMemset(d_C, 0, n_squared * sizeof(float)); // Llenamos de 0 la matriz de covarianza

    // Creacion stream
    cudaStream_t st[num_streams]; // Lista de identificadores de streams
    for (int i = 0; i < num_streams; i++)
        cudaStreamCreate(&st[i]); // Crea streams

    // Inicio de medicion
    cudaEventRecord(start);

    for (int i = 0; i < m; i++) {
        const int s_idx = i % num_streams; // Distribucion equitativa de  streams

        cudaMemcpyAsync(
            d_batch,
            h_dataset + static_cast<size_t>(i) * n,
            n * sizeof(float),
            cudaMemcpyHostToDevice,
            st[s_idx]
        );

        // Ejecucion de Kernels en el stream que corresponde
        kernel_add<<<(n + 255) / 256, 256, 0, st[s_idx]>>>(d_batch, d_mu, n);
        dim3 dimBlock(TILE_SIZE, TILE_SIZE);
        dim3 dimGrid((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

        kernel_covariance<<<dimGrid, dimBlock, 0, st[s_idx]>>>(d_batch, d_C, d_mu, n);
    }

    cudaDeviceSynchronize();

    // Fin de medicion
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    auto *out_C = static_cast<float *>(malloc(n_squared * sizeof(float)));
    cudaMemcpy(out_C, d_C, n_squared * sizeof(float), cudaMemcpyDeviceToHost);

    // Normalizacion
    for (size_t i = 0; i < n_squared; i++)
        out_C[i] /= static_cast<float>(m);

    // Reporte de información tecnica
    printf("\n--- Reporte de Ejecucion GPU ---\n");
    printf("Imagenes procesadas (m): %d\n", m);
    printf("Tamaño del vector (n): %d\n", n);
    printf("Streams utilizados: %d\n", num_streams);
    printf("Tiempo total de ejecucion (kernel + copia): %.3f ms\n", milliseconds);
    printf("Throughput estimado: %.2f img/sec\n", (m / (milliseconds / 1000.0)));
    printf("-------------------------------\n");

    // Limpieza
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    for (int i = 0; i < num_streams; i++)
        cudaStreamDestroy(st[i]);
    cudaFree(d_mu);
    cudaFree(d_C);
    cudaFree(d_batch);
}
