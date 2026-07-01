#pragma once

#include <cerrno>
#include <filesystem>
#include <iostream>

static constexpr int DEFAULT_WIDTH = 128;
static constexpr int DEFAULT_HEIGHT = 128;

struct Args {
    int width;
    int height;
    const std::filesystem::path dataset_dir;
};

inline Args parse_args(const int argc, const char *const *const argv) {
    namespace fs = std::filesystem;

    int width = DEFAULT_WIDTH;

    if (argc > 1) {
        errno = 0;
        char *end = nullptr;
        const char *arg = argv[1];
        width = static_cast<int>(std::strtol(arg, &end, 10));

        if (arg == end || errno == ERANGE || width <= 0) {
            std::cerr << argv[0] << ": invalid image width " << arg << std::endl;
            exit(1);
        }
    }

    int height = DEFAULT_HEIGHT;

    if (argc > 2) {
        errno = 0;
        char *end = nullptr;
        const char *arg = argv[2];
        height = static_cast<int>(std::strtol(arg, &end, 10));

        if (arg == end || errno == ERANGE || height <= 0) {
            std::cerr << argv[0] << ": invalid image height " << arg << std::endl;
            exit(1);
        }
    }

    fs::path dataset_dir = fs::path(__builtin_FILE()).parent_path().parent_path() / "dataset";

    if (argc > 3) {
        const char *arg = argv[3];
        dataset_dir = fs::absolute(arg);

        if (!fs::exists(dataset_dir) || !fs::is_directory(dataset_dir)) {
            std::cerr << argv[0] << ": invalid dataset directory " << arg << std::endl;
            exit(1);
        }
    }

    return {width, height, std::move(dataset_dir)};
}
