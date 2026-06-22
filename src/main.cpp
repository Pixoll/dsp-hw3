#define cimg_display 0

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "CImg.h"

using namespace cimg_library;

int main() {
    const auto dataset_path = std::filesystem::path(__builtin_FILE()).parent_path().parent_path() / "dataset";

    std::vector<std::filesystem::path> files;
    for (const auto &entry: std::filesystem::directory_iterator(dataset_path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".png") {
            files.push_back(entry.path());
        }
    }
    std::ranges::sort(files);

    const int num_images = static_cast<int>(files.size());
    constexpr int width = 192;
    constexpr int height = 192;
    // gray scale
    constexpr int channels = 1;
    // with size 192x192, n² is around 1.3 GB, with floats is around 5.4 GB
    constexpr int n = width * height * channels;

    for (const auto &file: files) {
        std::cout << "Processing " << file.stem() << "\n";

        CImg<unsigned char> img(file.c_str());

        // gray scale
        if (img.spectrum() > 1) {
            img = img.get_RGBtoYCbCr().get_channel(0);
        }

        // linear interpolation
        img.resize(width, height, 1, 1, 3);
    }

    return 0;
}
