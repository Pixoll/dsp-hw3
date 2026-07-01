#define cimg_display 0
#ifndef cimg_use_png
#define cimg_use_png
#endif

#include "args.hpp"
#include "CImg.h"
#include "experiment1.hpp"
#include "experiment2.hpp"
#include "preprocess.hpp"

using namespace cimg_library;

int main(const int argc, const char **argv) {
    const auto &[experiment, streams, width, height, dataset_dir] = parse_args(argc, argv);
    const auto &[h_dataset, n, m] = preprocess(dataset_dir, width, height);

    if (experiment == 1) {
        run_experiment1(h_dataset, m, n);
    } else {
        run_experiment2(h_dataset, m, n, streams);
    }

    free(h_dataset);
    return 0;
}
