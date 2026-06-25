# Distributed Systems and Parallelism - Homework 3

## Requirements

- Ubuntu 24.04
- CMake >= 3.20 (only for the preprocessing skeleton in `src/main.cpp`)
- CUDA Toolkit (`nvcc` in PATH) — built/tested with CUDA 13.3
- `libpng-dev` (CImg loads the `.png` dataset through libpng)

```bash
sudo apt-get install -y libpng-dev
```

## Experiment 1: Traditional CUDA implementation

Computes the covariance matrix (n x n) of the image set, on the GPU.
Sources live in `src/`, the loader header in `include/`.

### Compile

```bash
make exp1        # or: make all   (also builds the VRAM test)
```

This runs:

```bash
nvcc -arch=sm_86 -O3 -Iinclude -Dcimg_use_png src/exp1.cu src/loader.cpp -lpng -lz -o exp1
```

> `-arch=sm_86` targets Ampere (RTX 30xx). Change it to match your GPU.

### Run

```
./exp1 [width] [height] [dataset]
```

| Arg | Default | Meaning |
|-----|---------|---------|
| `width`   | `128`       | target image width (resized) |
| `height`  | `128`       | target image height |
| `dataset` | `./dataset` | folder with the `.png` images |

Channels is always 1 (grayscale); `n = width*height`. Pixels are normalized
to `[0,1]`. All images in the folder are loaded; timings are the average of
10 measured repetitions (after 1 discarded warm-up).

Examples:

```bash
./exp1                       # 128x128, ./dataset  (low-VRAM machines)
./exp1 192 192               # 192x192 (needs more VRAM for C ~5 GiB)
./exp1 192 192 /path/to/imgs # custom dataset folder
```

### VRAM viability test (Step 0)

```bash
make test_vram
./test_vram      # reports free/total VRAM and whether C fits at each size
```
