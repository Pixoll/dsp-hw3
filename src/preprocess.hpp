#pragma once

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <system_error>
#include <vector>

#include "CImg.h"

struct PreprocessResult {
    float *h_dataset;
    int n;
    int m;
};

inline PreprocessResult preprocess(const std::filesystem::path &dir, const int width, const int height) {
    namespace fs = std::filesystem;

    std::vector<fs::path> files;
    std::error_code ec;
    for (const auto &entry: fs::directory_iterator(dir, ec)) {
        if (entry.is_regular_file() && entry.path().extension() == ".png") {
            files.push_back(entry.path());
        }
    }

    if (ec || files.empty()) {
        std::cerr << "[PRE] no .png's found in '" << dir << "'" << std::endl;
        exit(1);
    }

    std::ranges::sort(files);

    const int m = static_cast<int>(files.size());
    const int n = width * height; // grayscale = 1 channel

    const auto data = static_cast<float *>(malloc(static_cast<size_t>(m) * n * sizeof(float)));
    if (!data) {
        const double megabytes = static_cast<double>(m) * n * sizeof(float) / 1000000.0;
        std::cerr << std::fixed << std::setprecision(2)
            << "[PRE] host malloc failed (" << megabytes << " MB)"
            << std::endl;
        exit(1);
    }

    for (size_t i = 0; i < m; ++i) {
        cimg_library::CImg<unsigned char> img(files[i].c_str());

        // set grayscale
        if (img.spectrum() > 1) {
            img = img.get_RGBtoYCbCr().get_channel(0);
        }
        // rescale with linear interpolation
        img.resize(width, height, 1, 1, 3);

        // flatten row-major
        const size_t base = i * n;
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                data[base + y * width + x] = static_cast<float>(img(x, y));
            }
        }
    }

    std::cout << "[PRE] " << m << " images loaded at " << width << "x" << height << "" << std::endl;
    return {data, n, m};
}
