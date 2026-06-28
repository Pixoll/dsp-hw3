# Distributed Systems and Parallelism - Homework 3

## Requirements

- Ubuntu 24.04
- CMake >= 3.20
- CUDA Toolkit (`nvcc` in PATH) — built/tested with CUDA 13.3
- `libpng-dev` (CImg loads the `.png` dataset through libpng)

```bash
sudo apt-get install -y libpng-dev
```

## Build

Single entry point: `src/main.cpp` preprocesses the dataset and dispatches to the
selected experiment (`run_exp1`, ...). Built with CMake (CUDA enabled).

```bash
./build.sh        # cmake -DCMAKE_BUILD_TYPE=Release -S . -B build && make
```

> The CUDA arch is set in `CMakeLists.txt` (`CMAKE_CUDA_ARCHITECTURES 86`,
> Ampere / RTX 30xx). Change it to match your GPU.

## Run

```
./build/dsp_hw3 [experiment] [width] [height] [dataset]
```

| Arg | Default | Meaning |
|-----|---------|---------|
| `experiment` | `1`         | `1` = traditional CUDA (`2` = streams, pending) |
| `width`      | `192`       | target image width (resized) |
| `height`     | `192`       | target image height |
| `dataset`    | `<repo>/dataset` | folder with the `.png` images |

Channels is always 1 (grayscale); `n = width*height`. All images in the folder are loaded; timings are the average of
10 measured repetitions (after 1 discarded warm-up).

Examples:

```bash
./run.sh                          # exp1, 128x128, default dataset
./build/dsp_hw3 1 192 192         # exp1 at 192x192 
./build/dsp_hw3 1 192 192 /imgs   # custom dataset folder
```

### VRAM viability test (Step 0)

`build.sh` also builds it:

```bash
./build/test_vram   # reports free/total VRAM and whether C fits at each size
```
