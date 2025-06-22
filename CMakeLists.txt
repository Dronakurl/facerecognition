cmake_minimum_required(VERSION 3.10)

project(
  FaceRecognition
  VERSION 0.1.0
  LANGUAGES CXX)

# For creating a deb file
set(CPACK_PROJECT_NAME "FaceRecognition")
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Face Recognition Library")
set(CPACK_PACKAGE_VENDOR "Dronakurl")
set(CPACK_PACKAGE_CONTACT "Dronakurl@discord")
set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_P)
set(CPACK_PROJECT_VERSION "${CPACK_PROJECT_NAME}-${CPACK_PROJECT_VERSION}")
include(CPack)

# Options
option(FACERECOGNITION_BUILD_EXAMPLES "Build example programs" ON)
option(FACERECOGNITION_USE_GLIB_LOGGING "Use GLib logging functions" OFF)

# Find required dependencies
find_package(OpenCV REQUIRED)

# Optional GLib dependency
if(FACERECOGNITION_USE_GLIB_LOGGING)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(GLIB REQUIRED glib-2.0)
else()
  message(STATUS "GLib logging is disabled")
endif()

# Create the library
add_library(facerecognition src/facerecognition.cpp)

# Add an alias for consistent naming
add_library(FaceRecognition::facerecognition ALIAS facerecognition)

# Include directories
target_include_directories(
  facerecognition
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${OpenCV_INCLUDE_DIRS})

# Link libraries
target_link_libraries(facerecognition PUBLIC ${OpenCV_LIBS})

# Optional GLib linking
if(FACERECOGNITION_USE_GLIB_LOGGING)
  target_link_libraries(facerecognition PRIVATE ${GLIB_LIBRARIES})
  target_include_directories(facerecognition PRIVATE ${GLIB_INCLUDE_DIRS})
  target_compile_definitions(facerecognition PRIVATE USE_GLIB_LOGGING)
endif()

# Compiler features and definitions
target_compile_features(facerecognition PUBLIC cxx_std_17)
target_compile_definitions(facerecognition
                           PUBLIC FACERECOGNITION_VERSION="${PROJECT_VERSION}")

# Set target properties
set_target_properties(
  facerecognition
  PROPERTIES VERSION ${PROJECT_VERSION}
             SOVERSION 1
             OUTPUT_NAME "facerecognition")

# Installation
include(GNUInstallDirs)

# Install the library
install(
  TARGETS facerecognition
  EXPORT FaceRecognitionTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# Install headers
install(
  DIRECTORY include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
  PATTERN "*.hpp")

# Export targets
install(
  EXPORT FaceRecognitionTargets
  FILE FaceRecognitionTargets.cmake
  NAMESPACE FaceRecognition::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FaceRecognition)

# Package configuration
include(CMakePackageConfigHelpers)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FaceRecognitionConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/FaceRecognitionConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FaceRecognition)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/FaceRecognitionConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/FaceRecognitionConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/FaceRecognitionConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/FaceRecognition)

# Examples
if(FACERECOGNITION_BUILD_EXAMPLES)
  message(STATUS "Building examples")
  if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/include/CLI11.hpp)
    message(STATUS "Downloading CLI11.hpp")
    file(DOWNLOAD
         https://github.com/CLIUtils/CLI11/releases/download/v2.5.0/CLI11.hpp
         ${CMAKE_CURRENT_SOURCE_DIR}/include/CLI11.hpp)
  else()
    message(STATUS "CLI11.hpp already found")
  endif()
  add_executable(facerecognition_example examples/facerecognition_cli.cpp)
  target_include_directories(
    facerecognition_example
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:include>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(facerecognition_example
                        PRIVATE FaceRecognition::facerecognition)
  install(TARGETS facerecognition_example
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

# Export from build tree (add this near the end)
export(
  EXPORT FaceRecognitionTargets
  FILE "${CMAKE_CURRENT_BINARY_DIR}/FaceRecognitionTargets.cmake"
  NAMESPACE FaceRecognition::)

# Register package in user's package registry
export(PACKAGE FaceRecognition)

# Create config file for build tree
configure_file(cmake/FaceRecognitionConfig.cmake.in
               "${CMAKE_CURRENT_BINARY_DIR}/FaceRecognitionConfig.cmake" @ONLY)
