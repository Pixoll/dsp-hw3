# Makefile - Tarea 3 Exp1 (CUDA tradicional)
# Requiere: CUDA Toolkit (nvcc en PATH) y libpng-dev (-lpng -lz).

NVCC  ?= nvcc
ARCH  ?= -arch=sm_86
INC    = -Iinclude
DEFS   = -Dcimg_use_png
LIBS   = -lpng -lz
SRC    = src

all: exp1 test_vram

exp1: $(SRC)/exp1.cu $(SRC)/loader.cpp include/loader.h
	$(NVCC) $(ARCH) -O3 $(INC) $(DEFS) $(SRC)/exp1.cu $(SRC)/loader.cpp $(LIBS) -o exp1

# Test de viabilidad de VRAM
test_vram: $(SRC)/test_vram.cu
	$(NVCC) $(ARCH) $(SRC)/test_vram.cu -o test_vram

clean:
	rm -f exp1 test_vram *.o

.PHONY: all clean
