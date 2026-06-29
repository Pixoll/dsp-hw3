#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define TILE_SIZE 16

// Kernel para promedio
__global__ void kernelSumar(const float *datos, float *suma_global, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) atomicAdd(&suma_global[idx], datos[idx]);
}

// Lo mismo pero para la covarianza
__global__ void kernelCov(const float *batch, float *C_final, float *mu, int n) {
    // Memoria compartida: es como una caché ultra rápida dentro del bloque
    __shared__ float sBatch[TILE_SIZE];

    int fila = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Solo cargamos el valor una vez en memoria compartida por cada hilo del bloque
    // Esto evita miles de accesos a la memoria global
    if (threadIdx.x < TILE_SIZE && fila < n) {
        sBatch[threadIdx.x] = batch[fila] - mu[fila];
    }
    __syncthreads(); // IMPORTANTE: Esperar a que todos los hilos del bloque carguen el dato

    if (fila < n && col < n) {
        // Ahora usamos los datos de la memoria compartida
        float val = sBatch[threadIdx.y] * sBatch[threadIdx.x];
        atomicAdd(&C_final[fila * n + col], val);
    }
}

void calcularCovarianzaGPU(float *h_data, int m, int n, int num_streams, float *out_C) {
    // Variables para medir tiempo
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Reservar memoria en GPU
    float *d_mu, *d_C, *d_batch;
    cudaMalloc(&d_mu, n * sizeof(float)); // Reserva de espacio para el promedio
    cudaMalloc(&d_C, (size_t)n * n * sizeof(float)); // Reserva de espacio para la matriz Covarianza
    cudaMalloc(&d_batch, n * sizeof(float)); // Reserva de espacio para 1 imagen

    cudaMemset(d_mu, 0, n * sizeof(float)); // LLenamos de 0 el vector promedio
    cudaMemset(d_C, 0, (size_t)n * n * sizeof(float)); // Llenamos de 0 la matriz de covarianza

    // Creacion stream
    cudaStream_t st[num_streams]; // Lista de identificadores de streams
    for(int i = 0; i < num_streams; i++) cudaStreamCreate(&st[i]); // Crea streams

    // Inicio de medicion
    cudaEventRecord(start);

    for (int i = 0; i < m; i++) {
        int s_idx = i % num_streams; // Distribucion equitativa de  streams

        cudaMemcpyAsync(d_batch, h_data + ((size_t)i * n), n * sizeof(float), cudaMemcpyHostToDevice, st[s_idx]);

        // Ejecucion de Kernels en el stream que corresponde
        kernelSumar<<<(n + 255) / 256, 256, 0, st[s_idx]>>>(d_batch, d_mu, n);
        dim3 dimBlock(TILE_SIZE, TILE_SIZE);
        dim3 dimGrid((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

        kernelCov<<<dimGrid, dimBlock, 0, st[s_idx]>>>(d_batch, d_C, d_mu, n);
    }

    cudaDeviceSynchronize();

    // Fin de medicion
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(out_C, d_C, (size_t)n * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Normalizacion
    for(size_t i = 0; i < (size_t)n * n; i++) out_C[i] /= m;

    // Reporte de informacion tecnica
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
    for(int i = 0; i < num_streams; i++) cudaStreamDestroy(st[i]);
    cudaFree(d_mu);
    cudaFree(d_C);
    cudaFree(d_batch);
}