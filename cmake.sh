cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17 \
    -DPSC_USE_ADIOS2=ON \
    -DCMAKE_PREFIX_PATH="/path/to/adios2/install" \
    -DUSE_CUDA=OFF \
    -DUSE_VPIC=OFF \
    ..
