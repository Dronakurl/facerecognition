<!-- markdownlint-disable MD013-->
# Face Recognition with OpenCV

A C++ face recognition library built on OpenCV's DNN module, featuring a simple way to arrange a database of persons to be recognized, which is watched for changes in real-time.

## Features

- **OpenCV DNN Models**: Uses [YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) for face detection and [SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) for recognition. (model files are downloaded once when the library is built)
- **Folder-based Person Database**: Organize faces in subfolders by person name
- **Automatic Database Reloading**: Watches database folder and reloads when changes are detected
- **Real-time Recognition**: Process images or video frames with bounding box visualization
- **CMake Package**: Integrate into your project.
- **Command line Example**: Simple command line example for a quick start

## Quick Start

### Requirements

OpenCV 4.x: Install with `sudo apt install libopencv-dev`.

### Set up person database

Put your photos of the persons who should be detected in a structure like this:

```txt
database/
├── person1/
│   ├── photo1.jpg
│   ├── foto2.jpg
│   └── photo3.jpg
├── person2/
│   ├── image1.png
│   └── image2.jpg
└── person3/
    └── face.jpg
```

### Run the command line example

build and run the [example](./examples/facerecognition_cli.cpp):

```bash
./examples/build_and_run_example.sh -i test_image.jpg -d ./database
```

## Integrate into your project

### build

```bash
rm -fr build && mkdir build && cd build
cmake .. 
make -j$(nproc)
cd ..
```

### cmake

Include this in your `CMakeLists.txt`:

```cmake
find_package(FaceRecognition REQUIRED)
target_link_libraries(your_app PRIVATE FaceRecognition::facerecognition)
```

### Sample code

```cpp
#include "facerecognition.hpp"

// Initialize face recognition
FaceRecognition faceRecognizer;
// Load person database
faceRecognizer.loadPersonsDB("/path/to/database");
// Enable automatic database watching (optional)
faceRecognizer.startWatching(5); // Check every 5 seconds
// Process an image
cv::Mat frame = cv::imread("test_image.jpg");
faceRecognizer.run(frame); 
```

## References

- The repo is based on this [opencv tutorial](https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html).
- The person library idea is based on [Deepface](https://github.com/serengil/deepface).

## Key Components

- **`FaceRecognition`**: Main class handling detection, recognition, and database management
- **`DetectedFace`**: Structure containing face information and features
- **Database Watcher**: Background thread monitoring folder changes for automatic reloading

The library automatically handles feature extraction, face alignment, and similarity matching using cosine distance, making it easy to build face recognition applications with minimal code.

## System wide installation (usually not needed)

To build and install the package:

```bash
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc)
sudo make install
```

Or create a linux package:

```bash
cpack -G DEB  # for .deb package
cpack -G RPM  # for .rpm package
```
