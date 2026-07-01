#define cimg_display 0
#ifndef cimg_use_png
#define cimg_use_png
#endif

#include <filesystem>

#include "args.hpp"
#include "CImg.h"
#include "experiment1.hpp"
#include "experiment2.hpp"
#include "preprocess.hpp"

using namespace cimg_library;

int main(const int argc, const char **argv) {
    namespace fs = std::filesystem;

    const auto &[experiment, streams, width, height, dataset_dir] = parse_args(argc, argv);
    const auto &[h_dataset, n, m] = preprocess(dataset_dir, width, height);
    const fs::path data_dir = fs::path(__builtin_FILE()).parent_path().parent_path() / "data";

    if (experiment == 1) {
        run_experiment1(h_dataset, m, n, width, height, data_dir);
    } else {
        run_experiment2(h_dataset, m, n, streams, width, height, data_dir);
    }

    free(h_dataset);
    return 0;
}
