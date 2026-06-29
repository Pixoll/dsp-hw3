# Distributed Systems and Parallelism - Homework 3

## Requirements

- Ubuntu 24.04
- CMake >= 3.20
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_local) >=
  13.3
- `libpng-dev` (CImg loads the `.png` dataset through `libpng`)
  ```bash
  sudo apt-get install -y libpng-dev
  ```

## Build

Built with CMake (CUDA enabled).

```bash
./build.sh
```

> The CUDA arch is set in `CMakeLists.txt` (`CMAKE_CUDA_ARCHITECTURES 86 89 120`, RTX 30xx 40xx 50xx).
> Change it to match your GPU.

### GPU memory viability test

Reports free/total GPU memory and whether matrix `C` fits.

```bash
./build/test_vram
```

## Run experiments

```
./build/dsp_hw3 <experiment> <streams> [width] [height] [dataset]
```

| Arg          | Default                     | Meaning                                    |
|--------------|-----------------------------|--------------------------------------------|
| `experiment` | required                    | `1` = traditional CUDA, `2` = CUDA streams |
| `streams`    | required (experiment = `2`) | number of CUDA streams for experiment 1    |
| `size`       | `128`                       | target image width                         |
| `size`       | `128`                       | target image height                        |
| `dataset`    | `./dataset`                 | folder with the `.png` images              |

All images in the folder are loaded into grayscale, and timings are the average result (after 1 discarded warm-up).

Examples:

```bash
./build/dsp_hw3 1                   # experiment 1 at 128x128
./build/dsp_hw3 2 8 192 192         # experiment 2 with 8 streams at 192x192
./build/dsp_hw3 1 0 192 192 /imgs   # experiment 1 at 192x192 with custom dataset folder
```
