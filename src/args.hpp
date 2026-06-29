#pragma once

#include <cerrno>
#include <filesystem>
#include <iostream>

static constexpr int DEFAULT_WIDTH = 128;
static constexpr int DEFAULT_HEIGHT = 128;

struct Args {
    int experiment;
    int width;
    int height;
    const std::filesystem::path dataset_dir;
};

inline Args parse_args(const int argc, const char *const *const argv) {
    namespace fs = std::filesystem;

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <experiment> [image_width] [image_height] [dataset_dir]" << std::endl;
        exit(1);
    }

    errno = 0;
    char *end = nullptr;
    const char *arg = argv[1];
    const int experiment = static_cast<int>(std::strtol(arg, &end, 10));

    if (arg == end || errno == ERANGE) {
        std::cerr << argv[0] << ": invalid experiment " << arg << std::endl;
        exit(1);
    }

    int width = DEFAULT_WIDTH;

    if (argc >= 3) {
        errno = 0;
        end = nullptr;
        arg = argv[2];
        width = static_cast<int>(std::strtol(arg, &end, 10));

        if (arg == end || errno == ERANGE || width <= 0) {
            std::cerr << argv[0] << ": invalid image width " << arg << std::endl;
            exit(1);
        }
    }

    int height = DEFAULT_HEIGHT;

    if (argc >= 4) {
        errno = 0;
        end = nullptr;
        arg = argv[3];
        height = static_cast<int>(std::strtol(arg, &end, 10));

        if (arg == end || errno == ERANGE || height <= 0) {
            std::cerr << argv[0] << ": invalid image height " << arg << std::endl;
            exit(1);
        }
    }

    fs::path dataset_dir = fs::path(__builtin_FILE()).parent_path().parent_path() / "dataset";

    if (argc >= 4) {
        dataset_dir = argv[3];

        if (!fs::exists(dataset_dir) || !fs::is_directory(dataset_dir)) {
            std::cerr << argv[0] << ": invalid dataset directory " << argv[3] << std::endl;
            exit(1);
        }
    }

    return {experiment, width, height, std::move(dataset_dir)};
}
