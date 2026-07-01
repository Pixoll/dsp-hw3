#pragma once

#include <filesystem>

void run_experiment1(
    const float *h_dataset,
    int m,
    int n,
    int width,
    int height,
    const std::filesystem::path &data_dir
);
