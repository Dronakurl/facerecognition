#!/bin/bash

REBUILD_FLAG=${REBUILD_FLAG:-"false"}

if [ ! -d "src" ]; then
  echo "Error: 'src' directory not found in the current path."
  echo "Run this script from the main directory "
  exit 1
fi

if [ "$REBUILD_FLAG" = "true" ]; then
  rm -fr build
fi
mkdir -p build && cd build || exit
cmake ..
bear -- make -j"$(nproc)"
cd ..

./build/facerecognition_example -i /app/media/testdata/IMG.jpg -d /app/media/db "$@"
