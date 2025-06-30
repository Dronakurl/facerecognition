#!/bin/bash

set -e

mkdir -p /opt/opencv
cd /opt/opencv || exit

if [ -z "$(ls -A /opt/opencv)" ]; then
  git clone https://github.com/opencv/opencv &&
    git -C opencv checkout 4.11.0 &&
    git clone https://github.com/opencv/opencv_contrib &&
    git -C opencv_contrib checkout 4.11.0 &&
    git clone https://github.com/opencv/opencv_extra &&
    git -C opencv_extra checkout 4.11.0 &&
    mkdir -p build
fi

mkdir -p /opt/opencv/build
cd /opt/opencv/build || exit
cmake -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX=/usr/local \
  -D WITH_CUDA=off \
  -D ENABLE_FAST_MATH=1 \
  -D CUDA_FAST_MATH=0 \
  -D WITH_CUBLAS=0 \
  -D OPENCV_DNN_CUDA=OFF \
  -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules \
  -D BUILD_opencv_cudacodec=OFF \
  ../opencv &&
  make -j"$(nproc)" &&
  make install
