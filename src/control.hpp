#pragma once

#include <iostream>
#include <omp.h>

inline double verify_cpu(const float *h_dataset, const float *h_C, const int m, const int n) {
    const int n_squared = n * n;
    const auto mu = static_cast<float *>(malloc(n * sizeof(float)));
    const auto Vc = static_cast<float *>(malloc(m * n * sizeof(float)));
    const auto VcT = static_cast<float *>(malloc(m * n * sizeof(float)));
    const auto Cref = static_cast<float *>(malloc(n_squared * sizeof(float)));
    if (!mu || !Vc || !VcT || !Cref) {
        std::cerr << "CPU malloc failed" << std::endl;
        exit(1);
    }

    const int threads = omp_get_max_threads();
    omp_set_num_threads(threads);

    std::cout << "Verifying on CPU with " << threads << " threads..." << std::endl;
    std::cout << "Calculating mu...";

    double start_time = omp_get_wtime();
    #pragma omp parallel for default(none) shared(h_dataset, mu, m, n)
    for (int j = 0; j < n; j++) {
        float s = 0.0f;
        for (int k = 0; k < m; k++) {
            s += h_dataset[k * n + j];
        }
        mu[j] = s / m;
    }
    double end_time = omp_get_wtime();

    std::cout << "\b\b\b, took " << end_time - start_time << " seconds" << std::endl;
    std::cout << "Calculating Vc...";

    start_time = omp_get_wtime();
    #pragma omp parallel for default(none) shared(h_dataset, mu, Vc, m, n)
    for (int k = 0; k < m; k++) {
        for (int j = 0; j < n; j++) {
            Vc[k * n + j] = h_dataset[k * n + j] - mu[j];
        }
    }
    end_time = omp_get_wtime();

    std::cout << "\b\b\b, took " << end_time - start_time << " seconds" << std::endl;
    std::cout << "Calculating VcT...";

    start_time = omp_get_wtime();
    #pragma omp parallel for collapse(2) default(none) shared(Vc, VcT, m, n)
    for (int k = 0; k < m; k++) {
        for (int j = 0; j < n; j++) {
            VcT[j * m + k] = Vc[k * n + j];
        }
    }
    end_time = omp_get_wtime();

    std::cout << "\b\b\b, took " << end_time - start_time << " seconds" << std::endl;
    std::cout << "Calculating Cref...";

    start_time = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic) default(none) shared(VcT, Cref, m, n)
    for (int a = 0; a < n; a++) {
        const float* row_a = VcT + a * m;
        for (int b = a; b < n; b++) {
            const float* row_b = VcT + b * m;
            float s = 0.0f;
            #pragma omp simd reduction(+:s)
            for (int k = 0; k < m; k++) {
                s += row_a[k] * row_b[k];
            }
            const float val = s / m;
            Cref[a * n + b] = val;
            Cref[b * n + a] = val;
        }
    }
    end_time = omp_get_wtime();

    std::cout << "\b\b\b, took " << end_time - start_time << " seconds" << std::endl;
    std::cout << "Calculating error...";

    double max_err = 0.0;
    double max_ref = 0.0;

    start_time = omp_get_wtime();
    #pragma omp parallel for default(none) shared(h_C, Cref, n_squared) reduction(max:max_err, max_ref)
    for (int i = 0; i < n_squared; i++) {
        const double d = fabsf(h_C[i] - Cref[i]);
        if (d > max_err) {
            max_err = d;
        }
        const double a = fabsf(Cref[i]);
        if (a > max_ref) {
            max_ref = a;
        }
    }
    end_time = omp_get_wtime();

    std::cout << "\b\b\b, took " << end_time - start_time << " seconds" << std::endl;

    free(mu);
    free(Vc);
    free(VcT);
    free(Cref);
    return max_ref > 0.0 ? max_err / max_ref : max_err;
}
