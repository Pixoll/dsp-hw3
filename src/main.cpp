#define cimg_display 0
#ifndef cimg_use_png
#define cimg_use_png
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "CImg.h"
#include "experiments.h"

using namespace cimg_library;
namespace fs = std::filesystem;

static float *preprocesar(const char *dir, int width, int height,
                          int *out_m, int *out_n)
{
    std::vector<fs::path> files;
    std::error_code ec;
    for (const auto &entry : fs::directory_iterator(dir, ec))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".png")
        {
            files.push_back(entry.path());
        }
    }
    if (ec || files.empty())
    {
        fprintf(stderr, "[PRE] no se encontraron .png en '%s'\n", dir);
        return nullptr;
    }
    std::ranges::sort(files);

    int m = static_cast<int>(files.size());
    int n = width * height; // escala de grises -> channels = 1
    *out_m = m;
    *out_n = n;

    float *data = (float *)malloc((size_t)m * n * sizeof(float));
    if (!data)
    {
        fprintf(stderr, "[PRE] malloc host fallo (%.2f MiB)\n",
                (double)m * n * sizeof(float) / 1048576.0);
        return nullptr;
    }

    for (int i = 0; i < m; ++i)
    {
        CImg<unsigned char> img(files[i].c_str());

        // A escala de grises: canal Y (luma) de YCbCr.
        if (img.spectrum() > 1)
        {
            img = img.get_RGBtoYCbCr().get_channel(0);
        }
        // Redimensionar a width x height con interpolacion lineal (=3).
        img.resize(width, height, 1, 1, 3);

        // Aplanar row-major. Valores en [0,255]
        size_t base = (size_t)i * n;
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                data[base + (size_t)y * width + x] = (float)img(x, y);
            }
        }
    }

    fprintf(stderr, "[PRE] %d imagenes %dx%d cargadas, gris, valores en [0,255]\n",
            m, width, height);
    return data;
}

int main(int argc, char **argv)
{
    int experimento = (argc > 1) ? atoi(argv[1]) : 1;
    int width = (argc > 2) ? atoi(argv[2]) : 128;
    int height = (argc > 3) ? atoi(argv[3]) : 128;

    std::string default_dir =
        (fs::path(__builtin_FILE()).parent_path().parent_path() / "dataset").string();
    const char *dir = (argc > 4) ? argv[4] : default_dir.c_str();

    if (width <= 0 || height <= 0)
    {
        fprintf(stderr, "Uso: %s [experimento] [width] [height] [dataset]\n", argv[0]);
        return EXIT_FAILURE;
    }

    int m = 0, n = 0;
    float *h_dataset = preprocesar(dir, width, height, &m, &n);
    if (!h_dataset)
    {
        fprintf(stderr, "No se cargaron imagenes. Abortando.\n");
        return EXIT_FAILURE;
    }

    int rc;
    switch (experimento)
    {
    case 1:
        rc = run_exp1(h_dataset, m, n);
        break;

    default:
        fprintf(stderr, "Experimento %d no implementado.\n", experimento);
        rc = EXIT_FAILURE;
    }

    free(h_dataset);
    return rc;
}
