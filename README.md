<!-- markdownlint-disable MD013-->
# Face Recognition Package

## How to build

```bash
rm -fr build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
bear -- make -j$(nproc)
cd ..
```

## Installation and Distribution

To build and install the package:

```bash
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc)
sudo make install
```

Or create a package:

```bash
cpack -G DEB  # for .deb package
cpack -G RPM  # for .rpm package
```
