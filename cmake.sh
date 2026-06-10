#!/bin/bash
# cmake.sh — Configura el build de PSC de manera sencilla
# Uso: mkdir -p build && cd build && bash ../cmake.sh

cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_CUDA=OFF \
    -DUSE_VPIC=OFF \
    -DPSC_USE_ADIOS2=OFF \
    -DBUILD_TESTING=OFF \
    ..
