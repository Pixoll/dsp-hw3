#pragma once

#include <cerrno>
#include <iostream>

static constexpr int DEFAULT_WIDTH = 128;
static constexpr int DEFAULT_HEIGHT = 128;

struct Args {
    int width;
    int height;
    int streams;
};

template <bool IsProfiler>
Args parse_args(const int argc, const char *const *const argv) {
    int arg_idx = 0;
    int streams = 1;

    if constexpr (IsProfiler) {
        if (argc <= ++arg_idx) {
            std::cerr << argv[0] << ": missing number of streams" << std::endl;
            exit(1);
        }

        errno = 0;
        char *end = nullptr;
        const char *arg = argv[arg_idx];
        streams = static_cast<int>(std::strtol(arg, &end, 10));

        if (arg == end || errno == ERANGE || streams <= 0) {
            std::cerr << argv[0] << ": invalid number of streams " << arg << std::endl;
            exit(1);
        }
    }

    int width = DEFAULT_WIDTH;

    if (argc > ++arg_idx) {
        errno = 0;
        char *end = nullptr;
        const char *arg = argv[arg_idx];
        width = static_cast<int>(std::strtol(arg, &end, 10));

        if (arg == end || errno == ERANGE || width <= 0) {
            std::cerr << argv[0] << ": invalid image width " << arg << std::endl;
            exit(1);
        }
    }

    int height = DEFAULT_HEIGHT;

    if (argc > ++arg_idx) {
        errno = 0;
        char *end = nullptr;
        const char *arg = argv[arg_idx];
        height = static_cast<int>(std::strtol(arg, &end, 10));

        if (arg == end || errno == ERANGE || height <= 0) {
            std::cerr << argv[0] << ": invalid image height " << arg << std::endl;
            exit(1);
        }
    }

    return {width, height, streams};
}
