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
./build/dsp_hw3 [width] [height]
```

| Arg      | Default | Meaning             |
|----------|---------|---------------------|
| `width`  | `128`   | target image width  |
| `height` | `128`   | target image height |

All images in the folder are loaded into grayscale, and timings are the average result (after 1 discarded warm-up).

Examples:

```bash
./build/dsp_hw3                 # run experiments at 128x128
./build/dsp_hw3 192 192         # run experiments at 192x192
```

### Run experiment 2 profiler (NVIDIA Nsight Systems)

Run a simplified version of the 2nd experiment (without timings calculations) against the NVIDIA profiler.

```
nsys profile --trace=cuda,nvtx,osrt -o report_s<streams> ./build/profile_exp2 <streams> [width] [height]
```

| Arg       | Default  | Meaning                       |
|-----------|----------|-------------------------------|
| `streams` | required | number of CUDA streams to use |
| `width`   | `128`    | target image width            |
| `height`  | `128`    | target image height           |

Examples:

```bash
# run experiment 2 at 128x128 with 4 streams
nsys profile --trace=cuda,nvtx,osrt -o report_s4 --force-overwrite=true ./build/profile_exp2 4
# run experiment 2 at 128x128 with 8 streams
nsys profile --trace=cuda,nvtx,osrt -o report_s8 --force-overwrite=true ./build/profile_exp2 8 192 192
```
