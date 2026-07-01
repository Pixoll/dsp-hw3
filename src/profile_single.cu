#define cimg_display 0
#ifndef cimg_use_png
#define cimg_use_png
#endif

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "args.hpp"
#include "benchmark.hpp"
#include "CImg.h"
#include "experiment2.hpp"
#include "preprocess.hpp"

using namespace cimg_library;

int main(const int argc, const char **argv) {
    namespace fs = std::filesystem;

    const auto &[width, height, dataset_dir] = parse_args(argc, argv);
    const auto &[h_dataset, n, m] = preprocess(dataset_dir, width, height);

    const fs::path scratch_csv =
        fs::path(__builtin_FILE()).parent_path().parent_path() / "data/profile_scratch.csv";
    std::ofstream out = open_measurements_csv(scratch_csv);

    int num_streams = 4;
    if (argc > 4) {
        num_streams = std::max(1, std::atoi(argv[4]));
    }

    std::cout << "[PROFILE] width=" << width << " height=" << height
              << " streams=" << num_streams << std::endl;

    run_experiment2(h_dataset, m, n, num_streams, width, height, out, -1);

    free(h_dataset);
    return 0;
}
