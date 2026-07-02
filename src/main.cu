#define cimg_display 0
#ifndef cimg_use_png
#define cimg_use_png
#endif

#include <array>
#include <filesystem>
#include <fstream>

#include "args.hpp"
#include "benchmark.hpp"
#include "CImg.h"
#include "experiment1.hpp"
#include "experiment2.hpp"
#include "preprocess.hpp"

using namespace cimg_library;

int main(const int argc, const char **argv) {
    namespace fs = std::filesystem;

    const fs::path repo_path = fs::path(__builtin_FILE()).parent_path().parent_path();
    const fs::path dataset_dir = repo_path/ "dataset";

    const auto &[width, height, _] = parse_args<false>(argc, argv);
    const auto &[h_dataset, n, m] = preprocess(dataset_dir, width, height);
    const fs::path data_path = fs::path(__builtin_FILE()).parent_path().parent_path() / "data/measurements.csv";
    constexpr std::array streams_list{1, 2, 4, 8, 16};

    std::ofstream out = open_measurements_csv(data_path);

    run_experiment1(h_dataset, m, n, width, height, out);

    double baseline_total_mean = -1;
    for (const auto &streams: streams_list) {
        const double total_mean = run_experiment2(h_dataset, m, n, streams, width, height, out, baseline_total_mean);
        if (streams == 1) {
            baseline_total_mean = total_mean;
        }
    }

    free(h_dataset);
    return 0;
}
