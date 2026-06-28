//  Experimento 1: Implementacion Tradicional en CUDA
//
//  Calcula la matriz de covarianza (n x n) de un conjunto de m imagenes en
//  escala de grises, centradas:
//      mu_j        = (1/m) * sum_k v_j^(k)
//      vbar_j^(k)  = v_j^(k) - mu_j
//      C_{jj'}     = (1/m) * sum_k vbar_j^(k) * vbar_{j'}^(k)
//                  = (1/m) * (V_centrada^T * V_centrada)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>

#include "experiments.h"

// Manejo de errores: toda llamada CUDA se envuelve en CUDA_CHECK. Ante un
// fallo (p.ej. cudaErrorMemoryAllocation) aborta con archivo, linea y la
// llamada concreta que fallo.
#define CUDA_CHECK(call)                                                \
    do                                                                  \
    {                                                                   \
        cudaError_t _e = (call);                                        \
        if (_e != cudaSuccess)                                          \
        {                                                               \
            fprintf(stderr, "[CUDA ERROR] %s:%d -> %s\n    en: %s\n",   \
                    __FILE__, __LINE__, cudaGetErrorString(_e), #call); \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

// Reserva en device con chequeo + reporte de bytes pedidos (out-of-memory es
// binario: o aloja o falla con cudaErrorMemoryAllocation).
static void *devMalloc(size_t bytes, const char *etiqueta)
{
    void *ptr = nullptr;
    cudaError_t e = cudaMalloc(&ptr, bytes);
    if (e != cudaSuccess)
    {
        fprintf(stderr,
                "[CUDA ERROR] cudaMalloc fallo para '%s': %.2f MiB (%zu bytes) -> %s\n",
                etiqueta, bytes / 1048576.0, bytes, cudaGetErrorString(e));
        exit(EXIT_FAILURE);
    }
    printf("  [VRAM] %-16s : %8.2f MiB  OK\n", etiqueta, bytes / 1048576.0);
    return ptr;
}

// Tamano del tile para la covarianza (memoria compartida). Tunable.
#define TILE 16

// Repeticiones medidas para promediar tiempos (ademas de 1 warm-up descartado).
#define NREPS 10

// ============================================================================
//  KERNEL 1: promedio por componente (reduccion sobre el lote de imagenes)
//  Firma CONGELADA (la reutiliza el Exp2): NO cambiar.
//
//  Layout row-major: la componente j de la imagen k esta en d_imgs[k*n + j].
//  Un hilo posee la componente j completa y acumula sobre las num_imgs
//  imagenes; escribe directamente el PROMEDIO mu_j en d_sum[j].
// ============================================================================
__global__ void kernelSumaComponentes(const float *d_imgs, float *d_sum,
                                      int num_imgs, int n)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x; // indice de componente
    if (j >= n)
        return;

    float acc = 0.0f;
    for (int k = 0; k < num_imgs; ++k)
    {
        acc += d_imgs[(size_t)k * n + j]; // coalescente entre hilos
    }
    d_sum[j] = acc / (float)num_imgs; // promedio mu_j
}

// ============================================================================
//  KERNEL 2: centrado elementwise IN-PLACE (resta el promedio).
//  Firma CONGELADA (la reutiliza el Exp2): NO cambiar.
// ============================================================================
__global__ void kernelCentrar(float *d_imgs, const float *d_mu,
                              int num_imgs, int n)
{
    size_t total = (size_t)num_imgs * n;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (size_t)gridDim.x * blockDim.x)
    {
        int j = idx % n; // componente -> mu_j
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
__global__ void kernelCovarianzaTiling(const float *d_imgs_centradas,
                                       float *d_C, int num_imgs, int n)
{
    __shared__ float sIzq[TILE][TILE]; // sub-tile de Vc^T (filas de C)
    __shared__ float sDer[TILE][TILE]; // sub-tile de Vc   (cols de C)

    int fila = blockIdx.y * TILE + threadIdx.y; // indice j  (0..n-1)
    int col = blockIdx.x * TILE + threadIdx.x;  // indice j' (0..n-1)

    float acc = 0.0f;

    for (int t0 = 0; t0 < num_imgs; t0 += TILE)
    {
        int kIzq = t0 + threadIdx.x; // imagen para sIzq
        int kDer = t0 + threadIdx.y; // imagen para sDer

        sIzq[threadIdx.y][threadIdx.x] =
            (fila < n && kIzq < num_imgs)
                ? d_imgs_centradas[(size_t)kIzq * n + fila]
                : 0.0f;

        sDer[threadIdx.y][threadIdx.x] =
            (col < n && kDer < num_imgs)
                ? d_imgs_centradas[(size_t)kDer * n + col]
                : 0.0f;

        __syncthreads();

#pragma unroll
        for (int t = 0; t < TILE; ++t)
        {
            acc += sIzq[threadIdx.y][t] * sDer[t][threadIdx.x];
        }
        __syncthreads();
    }

    if (fila < n && col < n)
    {
        d_C[(size_t)fila * n + col] = acc / (float)num_imgs;
    }
}

//  Referencia en CPU para validar los kernels (solo para n pequeño).
static double verificarCPU(const float *h_dataset, const float *h_C,
                           int m, int n)
{
    float *mu = (float *)malloc((size_t)n * sizeof(float));
    float *Vc = (float *)malloc((size_t)m * n * sizeof(float));
    float *Cref = (float *)malloc((size_t)n * n * sizeof(float));
    if (!mu || !Vc || !Cref)
    {
        fprintf(stderr, "malloc CPU falló\n");
        exit(1);
    }

    for (int j = 0; j < n; ++j)
    {
        double s = 0.0;
        for (int k = 0; k < m; ++k)
            s += h_dataset[(size_t)k * n + j];
        mu[j] = (float)(s / m);
    }
    for (int k = 0; k < m; ++k)
        for (int j = 0; j < n; ++j)
            Vc[(size_t)k * n + j] = h_dataset[(size_t)k * n + j] - mu[j];
    for (int a = 0; a < n; ++a)
        for (int b = 0; b < n; ++b)
        {
            double s = 0.0;
            for (int k = 0; k < m; ++k)
                s += (double)Vc[(size_t)k * n + a] * Vc[(size_t)k * n + b];
            Cref[(size_t)a * n + b] = (float)(s / m);
        }

    double maxErr = 0.0, maxRef = 0.0;
    for (size_t i = 0; i < (size_t)n * n; ++i)
    {
        double d = fabs((double)h_C[i] - (double)Cref[i]);
        if (d > maxErr)
            maxErr = d;
        double a = fabs((double)Cref[i]);
        if (a > maxRef)
            maxRef = a;
    }
    free(mu);
    free(Vc);
    free(Cref);
    return (maxRef > 0.0) ? maxErr / maxRef : maxErr;
}

// ============================================================================
//  Ejecuta una pasada completa (H->D, 3 kernels, D->H) y mide cada fase.
//  Se reutiliza para el warm-up (descartado) y para cada repeticion medida.
//  Cada pasada re-copia el dataset original (el centrado es in-place y
//  destruye d_imgs, por eso cada repeticion parte de datos frescos).
// ============================================================================
struct Tiempos
{
    float h2d, compute, d2h;
};

static Tiempos ejecutarPasada(const float *h_dataset, float *h_C,
                              float *d_imgs, float *d_mu, float *d_C,
                              size_t bytesData, size_t bytesC, int m, int n,
                              cudaEvent_t e0, cudaEvent_t e1, cudaEvent_t e2,
                              cudaEvent_t e3, cudaEvent_t e4, cudaEvent_t e5)
{
    int hilos = 256;
    int blkSuma = (n + hilos - 1) / hilos;
    size_t totalElem = (size_t)m * n;
    int blkCentrar = (int)((totalElem + hilos - 1) / hilos);
    if (blkCentrar > 65535)
        blkCentrar = 65535; // grid-stride cubre el resto
    dim3 blockCov(TILE, TILE);
    dim3 gridCov((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);

    // 1) Copia SINCRONA host->device (stream 0 por defecto)
    CUDA_CHECK(cudaEventRecord(e0));
    CUDA_CHECK(cudaMemcpy(d_imgs, h_dataset, bytesData, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(e1));

    // 2) Computo: tres kernels
    CUDA_CHECK(cudaEventRecord(e2));
    kernelSumaComponentes<<<blkSuma, hilos>>>(d_imgs, d_mu, m, n);
    kernelCentrar<<<blkCentrar, hilos>>>(d_imgs, d_mu, m, n);
    kernelCovarianzaTiling<<<gridCov, blockCov>>>(d_imgs, d_C, m, n);
    CUDA_CHECK(cudaEventRecord(e3));
    CUDA_CHECK(cudaGetLastError());

    // 3) Copia C device->host
    CUDA_CHECK(cudaEventRecord(e4));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(e5));
    CUDA_CHECK(cudaEventSynchronize(e5));

    Tiempos t;
    CUDA_CHECK(cudaEventElapsedTime(&t.h2d, e0, e1));
    CUDA_CHECK(cudaEventElapsedTime(&t.compute, e2, e3));
    CUDA_CHECK(cudaEventElapsedTime(&t.d2h, e4, e5));
    return t;
}

int run_exp1(const float *h_dataset, int m, int n)
{
    printf("=== Experimento 1: Covarianza en CUDA (tradicional) ===\n");
    printf("Numero de imagenes: m=%d   ->  n=%d   repeticiones medidas=%d\n\n",
           m, n, NREPS);

    size_t bytesData = (size_t)m * n * sizeof(float);
    size_t bytesC = (size_t)n * n * sizeof(float);
    size_t bytesMu = (size_t)n * sizeof(float);

    printf("Presupuesto VRAM: dataset=%.2f MiB + C=%.2f MiB + mu=%.2f MiB = %.2f MiB\n",
           bytesData / 1048576.0, bytesC / 1048576.0, bytesMu / 1048576.0,
           (bytesData + bytesC + bytesMu) / 1048576.0);

    size_t libre, total;
    CUDA_CHECK(cudaMemGetInfo(&libre, &total));
    printf("VRAM libre: %.2f MiB / total: %.2f MiB\n\n",
           libre / 1048576.0, total / 1048576.0);

    float *h_C = (float *)malloc(bytesC);
    if (!h_C)
    {
        fprintf(stderr, "malloc host de C falló (%.2f MiB)\n", bytesC / 1048576.0);
        return EXIT_FAILURE;
    }

    //  Device: reservas con chequeo
    printf("Reservas en device:\n");
    float *d_imgs = (float *)devMalloc(bytesData, "dataset");
    float *d_mu = (float *)devMalloc(bytesMu, "mu");
    float *d_C = (float *)devMalloc(bytesC, "C");
    printf("\n");

    // Eventos
    cudaEvent_t e0, e1, e2, e3, e4, e5;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventCreate(&e2));
    CUDA_CHECK(cudaEventCreate(&e3));
    CUDA_CHECK(cudaEventCreate(&e4));
    CUDA_CHECK(cudaEventCreate(&e5));

    // Warm-up (descartado): paga init de contexto / JIT
    ejecutarPasada(h_dataset, h_C, d_imgs, d_mu, d_C,
                   bytesData, bytesC, m, n, e0, e1, e2, e3, e4, e5);

    // N repeticiones medidas
    float sumH2D = 0, sumComp = 0, sumD2H = 0;
    float minH2D = FLT_MAX, minComp = FLT_MAX, minD2H = FLT_MAX;
    for (int r = 0; r < NREPS; ++r)
    {
        Tiempos t = ejecutarPasada(h_dataset, h_C, d_imgs, d_mu, d_C,
                                   bytesData, bytesC, m, n,
                                   e0, e1, e2, e3, e4, e5);
        sumH2D += t.h2d;
        sumComp += t.compute;
        sumD2H += t.d2h;
        if (t.h2d < minH2D)
            minH2D = t.h2d;
        if (t.compute < minComp)
            minComp = t.compute;
        if (t.d2h < minD2H)
            minD2H = t.d2h;
    }

    printf("=== Tiempos (promedio de %d reps | mínimo) ===\n", NREPS);
    printf("  Copia H->D (dataset) : %9.3f ms | min %9.3f ms\n", sumH2D / NREPS, minH2D);
    printf("  Computo (3 kernels)  : %9.3f ms | min %9.3f ms\n", sumComp / NREPS, minComp);
    printf("  Copia D->H (C)       : %9.3f ms | min %9.3f ms\n", sumD2H / NREPS, minD2H);
    printf("  TOTAL (promedio)     : %9.3f ms\n\n",
           (sumH2D + sumComp + sumD2H) / NREPS);

    // Validacion (solo n pequeño: la covarianza CPU es O(n^2 * m))
    if (n <= 1024)
    {
        double errRel = verificarCPU(h_dataset, h_C, m, n);
        printf("Verificación CPU: error relativo máx = %.3e  -> %s\n",
               errRel, (errRel < 1e-4) ? "OK" : "REVISAR");
    }
    else
    {
        printf("Verificación CPU omitida (n=%d > 1024; muy costosa en CPU).\n", n);
        printf("C[0..3]: %.6f %.6f %.6f %.6f\n", h_C[0], h_C[1], h_C[2], h_C[3]);
    }

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
    return EXIT_SUCCESS;
}