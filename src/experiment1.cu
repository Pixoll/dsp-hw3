#include "experiment1.hpp"

#include <cfloat>
#include <cstdlib>
#include <cuda_runtime.h>
#include <format>
#include <iomanip>
#include <iostream>

#include "control.hpp"

// Manejo de errores: toda llamada CUDA se envuelve en CUDA_CHECK. Ante un
// fallo (p.ej. cudaErrorMemoryAllocation) aborta con archivo, linea y la
// llamada concreta que fallo.
#define CUDA_CHECK(call)                                                \
    do                                                                  \
    {                                                                   \
        cudaError_t _e = (call);                                        \
        if (_e != cudaSuccess)                                          \
        {                                                               \
            fprintf(stderr, "[CUDA ERROR] %s:%d -> %s\n    at: %s\n",   \
                    __FILE__, __LINE__, cudaGetErrorString(_e), #call); \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

// Reserva en device con chequeo + reporte de bytes pedidos (out-of-memory es
// binario: o aloja o falla con cudaErrorMemoryAllocation).
static void *device_malloc(const size_t bytes, const char *label) {
    const double megabytes = static_cast<double>(bytes) / 1000000.0;
    void *ptr = nullptr;
    const cudaError_t e = cudaMalloc(&ptr, bytes);

    if (e != cudaSuccess) {
        const char *error_message = cudaGetErrorString(e);
        std::cerr << std::fixed << std::setprecision(2)
            << "[CUDA ERROR] cudaMalloc failed for '" << label << "': " << megabytes << " MB -> " << error_message
            << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << std::fixed << std::setprecision(2)
        << "  [VRAM] " << std::setw(16) << std::left << label << " : "
        << std::setw(8) << std::right << megabytes << " MB  OK"
        << std::endl;
    return ptr;
}

// Tamano del tile para la covarianza (memoria compartida). Tunable.
static constexpr int TILE = 16;

// Repeticiones medidas para promediar tiempos (ademas de 1 warm-up descartado).
static constexpr int NREPS = 10;

static constexpr int THREADS = 256;

// ============================================================================
//  KERNEL 1: promedio por componente (reduccion sobre el lote de imagenes)
//  Firma CONGELADA (la reutiliza el Exp2): NO cambiar.
//
//  Layout row-major: la componente j de la imagen k esta en d_imgs[k*n + j].
//  Un hilo posee la componente j completa y acumula sobre las num_imgs
//  imagenes; escribe directamente el PROMEDIO mu_j en d_sum[j].
// ============================================================================
__global__ void kernel_add_components(
    const float *d_imgs,
    float *d_sum,
    const int num_imgs,
    const int n
) {
    const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x; // indice de componente
    if (j >= n) {
        return;
    }

    float acc = 0.0f;
    for (size_t k = 0; k < num_imgs; ++k) {
        acc += d_imgs[k * n + j]; // coalescente entre hilos
    }
    d_sum[j] = acc / static_cast<float>(num_imgs); // promedio mu_j
}

// ============================================================================
//  KERNEL 2: centrado elementwise IN-PLACE (resta el promedio).
//  Firma CONGELADA (la reutiliza el Exp2): NO cambiar.
// ============================================================================
__global__ void kernel_center(
    float *d_imgs,
    const float *d_mu,
    const int num_imgs,
    const int n
) {
    const size_t total = static_cast<size_t>(num_imgs) * n;
    for (
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < total;
        idx += gridDim.x * blockDim.x
    ) {
        const size_t j = idx % n; // componente -> mu_j
        d_imgs[idx] -= d_mu[j];
    }
}

// ============================================================================
//  KERNEL 3: covarianza por multiplicacion matricial con TILING (shared mem).
//  Firma CONGELADA (la reutiliza el Exp2): NO cambiar.
//
//  C = (1/m) * (Vc^T * Vc), Vc = d_imgs_centradas (m x n),
//  Vc[k][j] = d_imgs_centradas[k*n + j]. C (n x n):
//      C[fila][col] = (1/m) * sum_k Vc[k][fila] * Vc[k][col]
//  Contraccion sobre m (imagenes); cada bloque calcula un tile TILE x TILE.
// ============================================================================
__global__ void kernel_covariance_tiling(
    const float *d_imgs_centradas,
    float *d_C,
    const int num_imgs,
    const int n
) {
    __shared__ float s_left[TILE][TILE]; // sub-tile de Vc^T (filas de C)
    __shared__ float s_right[TILE][TILE]; // sub-tile de Vc   (cols de C)

    const unsigned int fila = blockIdx.y * TILE + threadIdx.y; // indice j  (0..n-1)
    const unsigned int col = blockIdx.x * TILE + threadIdx.x; // indice j' (0..n-1)

    float acc = 0.0f;

    for (size_t t0 = 0; t0 < num_imgs; t0 += TILE) {
        const size_t k_left = t0 + threadIdx.x; // imagen para sIzq
        const size_t k_right = t0 + threadIdx.y; // imagen para sDer

        s_left[threadIdx.y][threadIdx.x] = fila < n && k_left < num_imgs
            ? d_imgs_centradas[k_left * n + fila]
            : 0.0f;

        s_right[threadIdx.y][threadIdx.x] = col < n && k_right < num_imgs
            ? d_imgs_centradas[k_right * n + col]
            : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int t = 0; t < TILE; ++t) {
            acc += s_left[threadIdx.y][t] * s_right[t][threadIdx.x];
        }
        __syncthreads();
    }

    if (fila < n && col < n) {
        d_C[fila * n + col] = acc / static_cast<float>(num_imgs);
    }
}

// ============================================================================
//  Ejecuta una pasada completa (H->D, 3 kernels, D->H) y mide cada fase.
//  Se reutiliza para el warm-up (descartado) y para cada repeticion medida.
//  Cada pasada re-copia el dataset original (el centrado es in-place y
//  destruye d_imgs, por eso cada repeticion parte de datos frescos).
// ============================================================================
struct Timings {
    float h2d, compute, d2h;
};

static Timings run_pass(
    const float *h_dataset,
    float *h_C,
    float *d_imgs,
    float *d_mu,
    float *d_C,
    const size_t bytesData,
    const size_t bytesC,
    const int m,
    const int n,
    const cudaEvent_t e0,
    const cudaEvent_t e1,
    const cudaEvent_t e2,
    const cudaEvent_t e3,
    const cudaEvent_t e4,
    const cudaEvent_t e5
) {
    const int block_sum = (n + THREADS - 1) / THREADS;
    const size_t total_elements = static_cast<size_t>(m) * n;
    size_t center_block = (total_elements + THREADS - 1) / THREADS;
    if (center_block > 65535) {
        center_block = 65535; // grid-stride cubre el resto
    }
    dim3 block_cov(TILE, TILE);
    dim3 grid_cov((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);

    // 1) Copia SINCRONA host->device (stream 0 por defecto)
    CUDA_CHECK(cudaEventRecord(e0));
    CUDA_CHECK(cudaMemcpy(d_imgs, h_dataset, bytesData, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(e1));

    // 2) Computo: tres kernels
    CUDA_CHECK(cudaEventRecord(e2));
    kernel_add_components<<<block_sum, THREADS>>>(d_imgs, d_mu, m, n);
    kernel_center<<<center_block, THREADS>>>(d_imgs, d_mu, m, n);
    kernel_covariance_tiling<<<grid_cov, block_cov>>>(d_imgs, d_C, m, n);
    CUDA_CHECK(cudaEventRecord(e3));
    CUDA_CHECK(cudaGetLastError());

    // 3) Copia C device->host
    CUDA_CHECK(cudaEventRecord(e4));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(e5));
    CUDA_CHECK(cudaEventSynchronize(e5));

    Timings t{};
    CUDA_CHECK(cudaEventElapsedTime(&t.h2d, e0, e1));
    CUDA_CHECK(cudaEventElapsedTime(&t.compute, e2, e3));
    CUDA_CHECK(cudaEventElapsedTime(&t.d2h, e4, e5));
    return t;
}

void run_experiment1(const float *h_dataset, const int m, const int n) {
    std::cout
        << "=== Experiment 1 ===\n"
        << "Number of images: m=" << m << " -> n=" << n << " repeticiones medidas=" << NREPS << "\n"
        << std::endl;

    const size_t bytes_data = static_cast<size_t>(m) * n * sizeof(float);
    const size_t bytes_c = static_cast<size_t>(n) * n * sizeof(float);
    const size_t bytes_mu = static_cast<size_t>(n) * sizeof(float);

    std::cout << std::fixed << std::setprecision(2)
        << "VRAM budget: dataset=" << static_cast<double>(bytes_data) / 1000000.0
        << " MB + C=" << static_cast<double>(bytes_c) / 1000000.0
        << " MB + mu=" << static_cast<double>(bytes_mu) / 1000000.0
        << " MB = " << static_cast<double>(bytes_data + bytes_c + bytes_mu) / 1000000.0
        << " MB" << std::endl;

    size_t free_memory, total;
    CUDA_CHECK(cudaMemGetInfo(&free_memory, &total));
    std::cout << std::fixed << std::setprecision(2)
        << "VRAM free: " << static_cast<double>(free_memory) / 1000000.0
        << " MB / total: " << static_cast<double>(total) / 1000000.0
        << " MB" << std::endl;

    auto *h_C = static_cast<float *>(malloc(bytes_c));
    if (!h_C) {
        std::cerr << std::fixed << std::setprecision(2)
            << "host malloc of C failed (" << static_cast<double>(bytes_c) / 1000000.0 << " MB)"
            << std::endl;
        exit(1);
    }

    //  Device: reservas con chequeo
    std::cout << "Device mallocs:" << std::endl;
    const auto d_imgs = static_cast<float *>(device_malloc(bytes_data, "dataset"));
    const auto d_mu = static_cast<float *>(device_malloc(bytes_mu, "mu"));
    const auto d_C = static_cast<float *>(device_malloc(bytes_c, "C"));
    std::cout << std::endl;

    // Eventos
    cudaEvent_t e0, e1, e2, e3, e4, e5;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventCreate(&e2));
    CUDA_CHECK(cudaEventCreate(&e3));
    CUDA_CHECK(cudaEventCreate(&e4));
    CUDA_CHECK(cudaEventCreate(&e5));

    // Warm-up (descartado): paga init de contexto / JIT
    run_pass(h_dataset, h_C, d_imgs, d_mu, d_C, bytes_data, bytes_c, m, n, e0, e1, e2, e3, e4, e5);

    // N repeticiones medidas
    float h2d_sum = 0, component_sum = 0, d2h_sum = 0;
    float h2d_min = FLT_MAX, component_min = FLT_MAX, d2h_min = FLT_MAX;
    for (int r = 0; r < NREPS; ++r) {
        auto [h2d, compute, d2h] = run_pass(
            h_dataset,
            h_C,
            d_imgs,
            d_mu,
            d_C,
            bytes_data,
            bytes_c,
            m,
            n,
            e0,
            e1,
            e2,
            e3,
            e4,
            e5
        );

        h2d_sum += h2d;
        component_sum += compute;
        d2h_sum += d2h;

        if (h2d < h2d_min) {
            h2d_min = h2d;
        }
        if (compute < component_min) {
            component_min = compute;
        }
        if (d2h < d2h_min) {
            d2h_min = d2h;
        }
    }

    std::cout << std::fixed << std::setprecision(3) << std::right
        << "=== Timings (average of " << NREPS << " reps | min) ===\n"
        << "  H->D copy (dataset) : " << std::setw(9) << h2d_sum / NREPS << " ms | min " << std::setw(9) << h2d_min
        << " ms\n"
        << "  Compute (3 kernels) : " << std::setw(9) << component_sum / NREPS << " ms | min " << std::setw(9)
        << component_min << " ms\n"
        << "  D->H copy (C)       : " << std::setw(9) << d2h_sum / NREPS << " ms | min " << std::setw(9) << d2h_min
        << " ms\n"
        << "  TOTAL (average)     : " << std::setw(9) << (h2d_sum + component_sum + d2h_sum) / NREPS << " ms\n"
        << std::endl;

    // error checking
    const double relative_error = verify_cpu(h_dataset, h_C, m, n);
    const char *message = relative_error < 1e-4 ? "OK" : "CHECK";
    std::cout << std::scientific << std::setprecision(3)
        << "CPU verification: max relative error = " << relative_error << "  -> " << message
        << std::endl;

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaEventDestroy(e2);
    cudaEventDestroy(e3);
    cudaEventDestroy(e4);
    cudaEventDestroy(e5);
    CUDA_CHECK(cudaFree(d_imgs));
    CUDA_CHECK(cudaFree(d_mu));
    CUDA_CHECK(cudaFree(d_C));
    free(h_C); // h_dataset lo libera el caller (main), es su dueño
}
