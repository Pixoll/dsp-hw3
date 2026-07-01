#include "experiment2.hpp"

#include <algorithm>
#include <cuda_runtime.h>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <vector>

#include "benchmark.hpp"
#include "control.hpp"

static constexpr int TILE_SIZE = 16;
static constexpr int IMAGES_PER_BATCH = 4;
static constexpr int THREADS_NORM = 256;

// Repeticiones medidas para promediar tiempos (ademas de 1 warm-up descartado).
static constexpr int NREPS = 32;

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

// ============================================================================
//  Ejecuta una pasada completa (fase 1: promedio batched+overlapped, fase 2:
//  covarianza batched+overlapped, copia final D->H de C) y mide cada fase.
//  Se reutiliza para el warm-up (descartado) y para cada repeticion medida.
//
//  d_mu / d_C se resetean a 0 al inicio de cada pasada porque los kernels
//  acumulan sobre ellos via atomicAdd (si no se resetean, las repeticiones
//  se irian sumando entre si).
//
//  phase1 = fase de promedio, phase2 = fase de covarianza, phase3 = D->H copy (C)
// ============================================================================
static PhaseSample run_pass(
    const float *h_dataset,
    float *out_C,
    float *d_mu,
    float *d_C,
    const std::vector<float *> &d_batch,
    std::vector<cudaStream_t> &streams,
    const int m,
    const int n,
    const int num_streams,
    const int num_batches,
    const size_t n_squared
) {
    cudaEvent_t ev_start, ev_after_phase1, ev_after_phase2, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_after_phase1);
    cudaEventCreate(&ev_after_phase2);
    cudaEventCreate(&ev_stop);

    // Cada pasada parte de acumuladores limpios (ver nota arriba).
    cudaMemset(d_mu, 0, n * sizeof(float));
    cudaMemset(d_C, 0, n_squared * sizeof(float));

    cudaEventRecord(ev_start);

    // --- Fase 1: promedio (transferencias + kernel, batched/overlapped) ---
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

    // --- Fase 2: covarianza (transferencias + kernel, batched/overlapped) ---
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

    // --- Copia final D->H de C ---
    cudaMemcpy(out_C, d_C, n_squared * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);

    float ms_phase1 = 0, ms_phase2 = 0, ms_copy_back = 0;
    cudaEventElapsedTime(&ms_phase1, ev_start, ev_after_phase1);
    cudaEventElapsedTime(&ms_phase2, ev_after_phase1, ev_after_phase2);
    cudaEventElapsedTime(&ms_copy_back, ev_after_phase2, ev_stop);

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_after_phase1);
    cudaEventDestroy(ev_after_phase2);
    cudaEventDestroy(ev_stop);

    return {
        .phase1 = ms_phase1,
        .phase2 = ms_phase2,
        .phase3 = ms_copy_back,
    };
}

void run_experiment2(
    const float *h_dataset,
    const int m,
    const int n,
    int num_streams,
    const int width,
    const int height,
    const std::filesystem::path &data_dir
) {
    num_streams = std::max(1, num_streams);
    const size_t n_squared = static_cast<size_t>(n) * n;

    std::cout
        << "=== Experiment 2 ===\n"
        << "Number of images: m=" << m << " -> n=" << n
        << " streams=" << num_streams << " repeticiones medidas=" << NREPS << "\n"
        << std::endl;

    const size_t dataset_bytes = static_cast<size_t>(m) * n * sizeof(float);
    cudaHostRegister(const_cast<float *>(h_dataset), dataset_bytes, cudaHostRegisterDefault);

    float *d_mu = nullptr, *d_C = nullptr;
    cudaMalloc(&d_mu, n * sizeof(float));
    cudaMalloc(&d_C, n_squared * sizeof(float));

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

    auto *out_C = static_cast<float *>(std::malloc(n_squared * sizeof(float)));
    if (!out_C) {
        std::cerr << "host malloc of C failed" << std::endl;
        exit(1);
    }

    // Warm-up (descartado): paga init de contexto / JIT
    std::cout << "Warming up..." << std::endl;
    run_pass(h_dataset, out_C, d_mu, d_C, d_batch, streams, m, n, num_streams, num_batches, n_squared);

    // N repeticiones medidas, acumuladas en el benchmark compartido
    Benchmark bm("experiment2", m, n, width, height, num_streams);

    for (int r = 0; r < NREPS; ++r) {
        std::cout << "Running pass #" << r + 1 << "..." << std::endl;
        bm.add_sample(
            run_pass(h_dataset, out_C, d_mu, d_C, d_batch, streams, m, n, num_streams, num_batches, n_squared)
        );
    }

    // error checking (sobre la ultima pasada medida)
    const double relative_error = verify_cpu(h_dataset, out_C, m, n);
    const char *message = relative_error < 1e-4 ? "OK" : "CHECK";
    std::cout << std::scientific << std::setprecision(3)
        << "CPU verification: max relative error = " << relative_error << "  -> " << message
        << std::endl;

    bm.set_correct(relative_error < 1e-4);

    // Speedup / eficiencia contra el baseline S=1.
    // Si num_streams > 1, se busca el CSV exp2_WxH_1.csv ya generado por una corrida previa con S=1.
    double baseline_total_mean = -1;
    if (num_streams > 1) {
        const auto baseline_path = exp2_csv_path(data_dir, width, height, 1);
        if (const auto baseline = read_total_mean(baseline_path)) {
            baseline_total_mean = *baseline;
        } else {
            std::cout
                << "[WARN] Could not find baseline S=1 en " << baseline_path.string()
                << ", execute with num_streams=1 first so we can calculate speedup/efficiency."
                << std::endl;
        }
    }

    const Result result = bm.finalize(baseline_total_mean);

    std::cout << std::fixed << std::setprecision(3) << std::right
        << "=== Timings (average of " << NREPS << " reps) ===\n"
        << "  Phase 1 (average)    : " << std::setw(9) << result.phase1.mean << " ms\n"
        << "  Phase 2 (covariance) : " << std::setw(9) << result.phase2.mean << " ms\n"
        << "  D->H copy (C)        : " << std::setw(9) << result.phase3.mean << " ms\n"
        << "  TOTAL (average)      : " << std::setw(9) << result.total.mean << " ms\n";

    if (result.speedup != -1) {
        std::cout
            << "  Speedup vs S=1       : " << std::setw(9) << result.speedup << "\n"
            << "  Efficiency           : " << std::setw(9) << result.efficiency << "\n";
    }
    std::cout << std::endl;

    const auto csv_path = exp2_csv_path(data_dir, width, height, num_streams);
    write_csv(csv_path, result);
    std::cout << "Measurements written to: " << csv_path.string() << std::endl;

    for (int s = 0; s < num_streams; ++s) {
        cudaStreamDestroy(streams[s]);
        cudaFree(d_batch[s]);
    }
    cudaFree(d_mu);
    cudaFree(d_C);

    cudaHostUnregister(const_cast<float *>(h_dataset));

    std::free(out_C);
}
