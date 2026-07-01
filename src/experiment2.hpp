#pragma once

#include <filesystem>

void run_experiment2(
    const float *h_dataset,
    int m,
    int n,
    int num_streams,
    int width,
    int height,
    const std::filesystem::path &data_dir
);
