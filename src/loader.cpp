#define cimg_display 0

#include "CImg.h"
#include "loader.h"

#include <algorithm>
#include <filesystem>
#include <vector>
#include <cstdio>
#include <cstdlib>

using namespace cimg_library;
namespace fs = std::filesystem;

int cargarDataset(const char *dir, int width, int height, int max_m,
                  float **out_data, int *out_n)
{
    // ---- Listar y ordenar los .png de la carpeta ---------------------------
    std::vector<fs::path> files;
    std::error_code ec;
    for (const auto &entry : fs::directory_iterator(dir, ec))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".png")
        {
            files.push_back(entry.path());
        }
    }
    if (ec)
    {
        fprintf(stderr, "[LOADER] no se pudo abrir la carpeta '%s'\n", dir);
        return 0;
    }
    if (files.empty())
    {
        fprintf(stderr, "[LOADER] no se encontraron .png en '%s'\n", dir);
        return 0;
    }
    std::sort(files.begin(), files.end());

    int m = static_cast<int>(files.size());
    if (max_m > 0 && max_m < m)
        m = max_m;          // tope opcional
    int n = width * height; // channels = 1
    *out_n = n;

    float *data = (float *)malloc((size_t)m * n * sizeof(float));
    if (!data)
    {
        fprintf(stderr, "[LOADER] malloc host falló (%.2f MiB)\n",
                (double)m * n * sizeof(float) / 1048576.0);
        return 0;
    }

    // ---- Procesar cada imagen ----------------------------------------------
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

        // Aplanar row-major y normalizar a [0,1].
        size_t base = (size_t)i * n;
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                data[base + (size_t)y * width + x] = img(x, y) / 255.0f;
            }
        }
    }

    *out_data = data;
    fprintf(stderr, "[LOADER] %d imágenes %dx%d cargadas, gris, normalizadas a [0,1]\n",
            m, width, height);
    return m;
}
