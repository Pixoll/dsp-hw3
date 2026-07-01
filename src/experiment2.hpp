#pragma once

#include "benchmark.hpp"

double run_experiment2(
    const float *h_dataset,
    int m,
    int n,
    int num_streams,
    int width,
    int height,
    std::ofstream &out,
    double baseline_total_mean
);
