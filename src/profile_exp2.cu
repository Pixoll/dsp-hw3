#define cimg_display 0
#ifndef cimg_use_png
#define cimg_use_png
#endif

#include <filesystem>
#include <fstream>
#include <iostream>

#include "args.hpp"
#include "CImg.h"
#include "experiment2.hpp"
#include "preprocess.hpp"

using namespace cimg_library;

int main(const int argc, const char **argv) {
    namespace fs = std::filesystem;

    const fs::path repo_path = fs::path(__builtin_FILE()).parent_path().parent_path();
    const fs::path dataset_dir = repo_path/ "dataset";
    const fs::path scratch_csv = repo_path / "data/profile_scratch.csv";

    const auto &[width, height, streams] = parse_args<true>(argc, argv);
    const auto &[h_dataset, n, m] = preprocess(dataset_dir, width, height);

    std::cout << "[PROFILE] width=" << width << " height=" << height << " streams=" << streams << std::endl;

    run_single_experiment2(h_dataset, m, n, streams);

    free(h_dataset);
    return 0;
}
