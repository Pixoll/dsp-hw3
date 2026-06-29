#include <filesystem>

#define cimg_display 0
#ifndef cimg_use_png
#define cimg_use_png
#endif

#include "args.hpp"
#include "CImg.h"
#include "experiment1.h"
#include "preprocess.hpp"

using namespace cimg_library;
namespace fs = std::filesystem;

int main(const int argc, const char **argv) {
    const auto &[experiment, width, height, dataset_dir] = parse_args(argc, argv);
    const auto &[h_dataset, n, m] = preprocess(dataset_dir, width, height);

    int exit_code;
    switch (experiment) {
        case 1:
            exit_code = run_experiment1(h_dataset, m, n);
            break;

        default:
            std::cerr << "Unknown experiment: " << experiment << std::endl;
            exit_code = 1;
    }

    free(h_dataset);
    return exit_code;
}
