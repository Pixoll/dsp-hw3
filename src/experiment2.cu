#include "experiment2.hpp"

#include <algorithm>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <vector>

#include "control.hpp"

static constexpr int TILE_SIZE = 16;
static constexpr int IMAGES_PER_BATCH = 4;
static constexpr int THREADS_NORM = 256;

__global__ void kernel_batch_sum(const float *d_batch, float *d_mu_accum, const int n, const int batch_size) {
    const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= static_cast<unsigned int>(n)) return;

    float local_sum = 0.0f;
    for (int b = 0; b < batch_size; ++b) {
        local_sum += d_batch[static_cast<size_t>(b) * n + j];
    }
    atomicAdd(&d_mu_accum[j], local_sum);
}

__global__ void kernel_normalize_mu(float *d_mu, const int n, const float m) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < static_cast<unsigned int>(n)) d_mu[idx] /= m;
}

__global__ void kernel_covariance_batch(
    const float *d_batch,
    float *d_C,
    const float *d_mu,
    const int n,
    const int batch_size
) {
    __shared__ float s_row[TILE_SIZE];
    __shared__ float s_col[TILE_SIZE];

    const unsigned int fila = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    for (int b = 0; b < batch_size; ++b) {
        if (threadIdx.x == 0 && fila < static_cast<unsigned int>(n)) {
            s_row[threadIdx.y] = d_batch[static_cast<size_t>(b) * n + fila] - d_mu[fila];
        }
        if (threadIdx.y == 0 && col < static_cast<unsigned int>(n)) {
            s_col[threadIdx.x] = d_batch[static_cast<size_t>(b) * n + col] - d_mu[col];
        }
        __syncthreads();
        if (fila < static_cast<unsigned int>(n) && col < static_cast<unsigned int>(n)) {
            atomicAdd(
                &d_C[static_cast<size_t>(fila) * n + col],
                s_row[threadIdx.y] * s_col[threadIdx.x]
            );
        }
        __syncthreads();
    }
}

__global__ void kernel_normalize_C(float *d_C, const size_t n_squared, const float m) {
    for (size_t idx = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
         idx < n_squared;
         idx += static_cast<size_t>(gridDim.x) * blockDim.x) {
        d_C[idx] /= m;
    }
}

void run_experiment2(const float *h_dataset, const int m, const int n, int num_streams) {
    num_streams = std::max(1, num_streams);
    const size_t n_squared = static_cast<size_t>(n) * n;

    const size_t dataset_bytes = static_cast<size_t>(m) * n * sizeof(float);
    cudaHostRegister(const_cast<float *>(h_dataset), dataset_bytes, cudaHostRegisterDefault);

    float *d_mu = nullptr, *d_C = nullptr;
    cudaMalloc(&d_mu, n * sizeof(float));
    cudaMalloc(&d_C, n_squared * sizeof(float));
    cudaMemset(d_mu, 0, n * sizeof(float));
    cudaMemset(d_C, 0, n_squared * sizeof(float));

    std::vector<float *> d_batch(num_streams, nullptr);
    const size_t batch_buffer_floats = static_cast<size_t>(IMAGES_PER_BATCH) * n;
    for (int s = 0; s < num_streams; ++s) {
        cudaMalloc(&d_batch[s], batch_buffer_floats * sizeof(float));
    }

    std::vector<cudaStream_t> streams(num_streams);
    for (int s = 0; s < num_streams; ++s) {
        cudaStreamCreate(&streams[s]);
    }

    const int num_batches = (m + IMAGES_PER_BATCH - 1) / IMAGES_PER_BATCH;

    cudaEvent_t ev_start, ev_after_phase1, ev_after_phase2, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_after_phase1);
    cudaEventCreate(&ev_after_phase2);
    cudaEventCreate(&ev_stop);

    cudaEventRecord(ev_start);

    for (int b = 0; b < num_batches; ++b) {
        const int stream_id = b % num_streams;
        const int offset_img = b * IMAGES_PER_BATCH;
        const int actual_batch_size = std::min(IMAGES_PER_BATCH, m - offset_img);
        const size_t bytes = static_cast<size_t>(actual_batch_size) * n * sizeof(float);

        cudaMemcpyAsync(
            d_batch[stream_id],
            h_dataset + static_cast<size_t>(offset_img) * n,
            bytes,
            cudaMemcpyHostToDevice,
            streams[stream_id]
        );

        kernel_batch_sum<<<(n + THREADS_NORM - 1) / THREADS_NORM, THREADS_NORM, 0,
            streams[stream_id]>>>(
                d_batch[stream_id],
                d_mu,
                n,
                actual_batch_size
            );
    }
    cudaDeviceSynchronize();

    kernel_normalize_mu<<<(n + THREADS_NORM - 1) / THREADS_NORM, THREADS_NORM>>>(
        d_mu,
        n,
        static_cast<float>(m)
    );
    cudaDeviceSynchronize();
    cudaEventRecord(ev_after_phase1);

    constexpr dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    const dim3 dimGrid((n + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

    for (int b = 0; b < num_batches; ++b) {
        const int stream_id = b % num_streams;
        const int offset_img = b * IMAGES_PER_BATCH;
        const int actual_batch_size = std::min(IMAGES_PER_BATCH, m - offset_img);
        const size_t bytes = static_cast<size_t>(actual_batch_size) * n * sizeof(float);

        cudaMemcpyAsync(
            d_batch[stream_id],
            h_dataset + static_cast<size_t>(offset_img) * n,
            bytes,
            cudaMemcpyHostToDevice,
            streams[stream_id]
        );

        kernel_covariance_batch<<<dimGrid, dimBlock, 0, streams[stream_id]>>>(
            d_batch[stream_id],
            d_C,
            d_mu,
            n,
            actual_batch_size
        );
    }
    cudaDeviceSynchronize();
    cudaEventRecord(ev_after_phase2);

    const int blocks_norm = static_cast<int>(
        std::min<size_t>(65535, (n_squared + THREADS_NORM - 1) / THREADS_NORM));
    kernel_normalize_C<<<blocks_norm, THREADS_NORM>>>(d_C, n_squared, static_cast<float>(m));
    cudaDeviceSynchronize();

    auto *out_C = static_cast<float *>(std::malloc(n_squared * sizeof(float)));
    cudaMemcpy(out_C, d_C, n_squared * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);

    float ms_total = 0, ms_phase1 = 0, ms_phase2 = 0, ms_copy_back = 0;
    cudaEventElapsedTime(&ms_total, ev_start, ev_stop);
    cudaEventElapsedTime(&ms_phase1, ev_start, ev_after_phase1);
    cudaEventElapsedTime(&ms_phase2, ev_after_phase1, ev_after_phase2);
    cudaEventElapsedTime(&ms_copy_back, ev_after_phase2, ev_stop);

    std::printf("\n--- Reporte de Ejecucion GPU (Experimento 2) ---\n");
    std::printf("Imagenes procesadas (m):      %d\n", m);
    std::printf("Tamaño del vector (n):        %d\n", n);
    std::printf("Streams utilizados (S):       %d\n", num_streams);
    std::printf("Imagenes por batch:           %d (num_batches=%d)\n", IMAGES_PER_BATCH, num_batches);
    std::printf("Tiempo Fase 1 (promedio):     %.3f ms\n", ms_phase1);
    std::printf("Tiempo Fase 2 (covarianza):   %.3f ms\n", ms_phase2);
    std::printf("Tiempo copia C (D2H):         %.3f ms\n", ms_copy_back);
    std::printf("Tiempo total de ejecucion:    %.3f ms\n", ms_total);
    std::printf("Throughput estimado:          %.2f img/seg\n", m / (ms_total / 1000.0));
    std::printf("-------------------------------------------------\n");

    // error checking
    const double relative_error = verify_cpu(h_dataset, out_C, m, n);
    const char *message = relative_error < 1e-4 ? "OK" : "CHECK";
    std::cout << std::scientific << std::setprecision(3)
        << "CPU verification: max relative error = " << relative_error << "  -> " << message
        << std::endl;

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_after_phase1);
    cudaEventDestroy(ev_after_phase2);
    cudaEventDestroy(ev_stop);

    for (int s = 0; s < num_streams; ++s) {
        cudaStreamDestroy(streams[s]);
        cudaFree(d_batch[s]);
    }
    cudaFree(d_mu);
    cudaFree(d_C);

    cudaHostUnregister(const_cast<float *>(h_dataset));

    std::free(out_C);
}
