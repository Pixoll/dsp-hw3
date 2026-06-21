#!/bin/bash -x

mkdir -p build
cmake -DCMAKE_BUILD_TYPE=Release -S . -B build
cd build
make
