#include "facerecognition.hpp"
#include "helper.hpp"
#include <cassert>
#include <filesystem>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/objdetect/face.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#define scoreThreshold 0.7
#define nmsThreshold 0.3
#define topK 5000

FaceRecognition::FaceRecognition(const std::string &fdModelPath, const std::string &frModelPath,
                                 int maxSize) {
  FR_DEBUG("Initializing face recognition");
  FR_DEBUG("Testing face detection model file exists: %s", fdModelPath.c_str());
  assert(std::filesystem::exists(fdModelPath));
  FR_DEBUG("Testing face recognition model file exists: %s", frModelPath.c_str());
  assert(std::filesystem::exists(frModelPath));
  this->detector = FaceDetectorYN::create(fdModelPath, "", Size(400, 400), scoreThreshold,
                                          nmsThreshold, topK, 0, 0);
  this->face_recognizer = FaceRecognizerSF::create(frModelPath, "");
  this->maxSize = maxSize;
}

FaceRecognition::~FaceRecognition() { stopWatching(); }

void FaceRecognition::startWatching(int check_interval_seconds) {
  if (dbPath.empty()) {
    FR_ERROR("Cannot start watching: no database path set");
    return;
  }

  if (watcherRunning.load()) {
    FR_DEBUG("Watcher already running");
    return;
  }

  checkInterval = check_interval_seconds;
  lastModTime = getLatestModTime(dbPath);
  watcherRunning.store(true);
  watcherThread = std::thread(&FaceRecognition::watcherThreadFunc, this);
  FR_DEBUG("Started watching database folder: %s", dbPath.c_str());
}

void FaceRecognition::stopWatching() {
  if (watcherRunning.load()) {
    watcherRunning.store(false);
    if (watcherThread.joinable()) {
      watcherThread.join();
    }
    FR_DEBUG("Stopped watching database folder");
  }
}

void FaceRecognition::watcherThreadFunc() {
  while (watcherRunning.load()) {
    try {
      auto currentModTime = getLatestModTime(dbPath);
      if (currentModTime > lastModTime) {
        FR_DEBUG("Database folder changed, reloading...");
        lastModTime = currentModTime;
        loadPersonsDB(dbPath, true, false);
      }
    } catch (const std::exception &e) {
      FR_DEBUG("Error checking database folder: %s", e.what());
    }

    std::this_thread::sleep_for(std::chrono::seconds(checkInterval));
  }
}

filesystem::file_time_type FaceRecognition::getLatestModTime(const filesystem::path &path) {
  filesystem::file_time_type latestTime = filesystem::file_time_type::min();

  try {
    for (const auto &entry : filesystem::recursive_directory_iterator(path)) {
      if (entry.is_regular_file()) {
        auto fileTime = entry.last_write_time();
        if (fileTime > latestTime) {
          latestTime = fileTime;
        }
      }
    }
  } catch (const std::exception &e) {
    FR_DEBUG("Error accessing directory %s: %s", path.c_str(), e.what());
  }

  return latestTime;
}

void FaceRecognition::visualize(Mat &input, int frame, Mat &faces, int thickness) {
  if (frame >= 0)
    cout << "Frame " << frame << ", ";
  for (int i = 0; i < faces.rows; i++) {
    ostringstream oss;
    oss << "Face " << i << ", top-left coordinates: (" << faces.at<float>(i, 0) << ", "
        << faces.at<float>(i, 1) << "), "
        << "box width: " << faces.at<float>(i, 2) << ", box height: " << faces.at<float>(i, 3)
        << ", score: " << format("%.2f", faces.at<float>(i, 14));
    FR_DEBUG("%s", oss.str().c_str());

    rectangle(input,
              Rect2i(int(faces.at<float>(i, 0)), int(faces.at<float>(i, 1)),
                     int(faces.at<float>(i, 2)), int(faces.at<float>(i, 3))),
              Scalar(0, 255, 0), thickness);
    circle(input, Point2i(int(faces.at<float>(i, 4)), int(faces.at<float>(i, 5))), 2,
           Scalar(255, 0, 0), thickness);
    circle(input, Point2i(int(faces.at<float>(i, 6)), int(faces.at<float>(i, 7))), 2,
           Scalar(0, 0, 255), thickness);
    circle(input, Point2i(int(faces.at<float>(i, 8)), int(faces.at<float>(i, 9))), 2,
           Scalar(0, 255, 0), thickness);
    circle(input, Point2i(int(faces.at<float>(i, 10)), int(faces.at<float>(i, 11))), 2,
           Scalar(255, 0, 255), thickness);
    circle(input, Point2i(int(faces.at<float>(i, 12)), int(faces.at<float>(i, 13))), 2,
           Scalar(0, 255, 255), thickness);
  }
}

void FaceRecognition::resizeFrame(Mat &frame, bool keepAspectRatio) {
  // No Resizing requested
  if (maxSize <= 0)
    return;
  if (frame.empty()) {
    FR_WARNING("Frame is empty or invalid");
    return;
  }
  if (keepAspectRatio) {
    if (frame.cols > maxSize || frame.rows > maxSize) {
      int max_dim = max(frame.cols, frame.rows);
      float scale = maxSize / (float)max_dim;
      resize(frame, frame, Size(), scale, scale);
    }
  } else {
    resize(frame, frame, Size(maxSize, maxSize));
  }
}

vector<DetectedFace> FaceRecognition::extractFeatures(Mat &frame) {
  if (!detector) {
    FR_ERROR("Detector is null");
    return {};
  }
  if (frame.empty()) {
    FR_ERROR("Frame is empty or invalid");
    return {};
  }
  Size originalSize = frame.size();
  resizeFrame(frame, true);
  FR_DEBUG("Frame size: %d x %d", frame.cols, frame.rows);
  // FR_DEBUG("Detector input size: %d x %d", detector->getInputSize().width,
  //          detector->getInputSize().height);
  detector->setInputSize(frame.size());
  Mat faces;
  detector->detect(frame, faces);
  // FR_DEBUG("Found %d faces", faces.rows);
  if (faces.rows <= 0) {
    FR_WARNING("Cannot find any faces");
  }
  vector<DetectedFace> detfaces;
  for (int i = 0; i < faces.rows; i++) {
    Mat aligned_img;
    face_recognizer->alignCrop(frame, faces.row(i), aligned_img);
    Mat feature;
    face_recognizer->feature(aligned_img, feature);
    detfaces.push_back(
        DetectedFace{"Unknown", faces.row(i).clone(), feature.clone(), originalSize});
  }
  return detfaces;
}

MatchResults FaceRecognition::findBestMatch(const Mat &faceFeature, float threshold) {
  MatchResult bestmatch = {"Unknown", 0.0};
  vector<MatchResult> results;

  for (const auto &pair : featuresMap) {
    const string &personName = pair.first;
    const vector<Mat> features = pair.second;
    for (Mat feat : features) {
      double score = face_recognizer->match(faceFeature, feat, FaceRecognizerSF::FR_COSINE);
      results.push_back({personName, float(score)});
      FR_DEBUG("Person %s, score: %f", personName.c_str(), score);
      if ((score > bestmatch.score) && (score > threshold)) {
        bestmatch = MatchResult{personName, float(score)};
      }
    }
  }

  return MatchResults{results, bestmatch};
}

void FaceRecognition::loadPersonsDB(filesystem::path persondb_folder, bool force, bool visualize) {
  if (dbPath.empty()) {
    FR_DEBUG("Loading personsDB from %s", persondb_folder.c_str());
    this->isDBLoaded = NOT_LOADED;
  } else if (dbPath != persondb_folder) {
    FR_DEBUG("Database path changed, reloading...");
    this->isDBLoaded = NOT_LOADED;
  }
  this->dbPath = persondb_folder;

  if (this->isDBLoaded == LOADED && !force) {
    FR_DEBUG("load status, %s", getLoadStatusString(this->isDBLoaded).c_str());
    FR_DEBUG("force, %s", force ? "true" : "false");
    FR_DEBUG("PersonsDB already loaded, skipping");
    return;
  }
  this->isDBLoaded = LOADING;
  FR_DEBUG("Loading personsDB from %s", persondb_folder.c_str());
  this->featuresMap.clear();
  // Iterate over all folders
  for (auto &p : filesystem::directory_iterator(persondb_folder)) {
    if (p.is_directory()) {
      string personName = p.path().filename().string();
      vector<Mat> features;
      FR_DEBUG("Loading person: %s", personName.c_str());
      for (auto &imgPath : filesystem::directory_iterator(p.path())) {
        if (!imgPath.is_directory()) {
          FR_DEBUG("Loading image: %s for person %s", imgPath.path().c_str(), personName.c_str());
          // If "_visualize" is part of the filename, skip it
          if (imgPath.path().filename().string().find("_visualize") != string::npos) {
            continue;
          }
          Mat img = imread(imgPath.path().string());
          if (img.empty()) {
            FR_ERROR("Cannot read image: %s", imgPath.path().c_str());
            continue;
          }
          for (const DetectedFace &detectedFace : extractFeatures(img)) {
            features.push_back(detectedFace.feature);
          }
          if (visualize) {
            filesystem::path original_path = imgPath.path();
            string stem = original_path.stem().string();
            string extension = original_path.extension().string();
            filesystem::path visualize_path =
                original_path.parent_path() / (stem + "_visualize" + extension);
            imwrite(visualize_path.string(), img);
          }
        } else {
          FR_ERROR("Unexpected sub-directory: %s", imgPath.path().c_str());
        }
      }
      featuresMap[personName] = features;
    } else {
      FR_ERROR("Unexpected file: %s", p.path().c_str());
    }
  }
  this->isDBLoaded = LOADED;
}

void FaceRecognition::annotate_with_name(Mat &frame, const DetectedFace &face) {

  // Extract the relevant information from the face object
  std::string name = face.name;
  FR_DEBUG("Annotating face with name: %s", face.name.c_str());
  Rect2i bbox = face.bbox();
  FR_DEBUG("bbox: %d %d %d %d", bbox.x, bbox.y, bbox.width, bbox.height);

  // Determine Text parameters
  int font = FONT_HERSHEY_SIMPLEX;
  double fontScale = 0.8;
  int thickness = 2;
  int baseline = 0;
  Size textSize = getTextSize(name, font, fontScale, thickness, &baseline);
  int text_x = bbox.x + (bbox.width - textSize.width) / 2;
  int text_y = std::max(bbox.y - textSize.height - 5, 0);

  // Draw the rectangle
  Rect bgRect(text_x - 2, text_y - 2, textSize.width + 4, textSize.height + 4);
  rectangle(frame, bgRect, Scalar(0, 0, 0), FILLED);
  putText(frame, name, Point(text_x, text_y + textSize.height), font, fontScale,
          Scalar(255, 255, 255), thickness);
}

vector<MatchResult> FaceRecognition::run(Mat &frame, float threshold, bool visualize) {
  vector<DetectedFace> det_faces;
  if (!visualize) {
    Mat frame_copy = frame.clone();
    det_faces = extractFeatures(frame_copy);
  } else {
    det_faces = extractFeatures(frame);
  }
  vector<MatchResult> results;
  int i = 1;
  for (DetectedFace &face : det_faces) {
    MatchResult best = findBestMatch(face.feature, threshold).bestmatch;
    face.name = best.name;
    FR_INFO("Face %d best match: %s", i, face.name.c_str());
    results.push_back(best);
    i++;
    if (visualize) {
      this->visualize(frame, -1, face.facedetect);
      annotate_with_name(frame, face);
    }
  }
  return results;
}

MatchResult FaceRecognition::run_one_face(Mat frame, float threshold, bool visualize) {
  vector<MatchResult> results = run(frame, threshold, visualize);
  if (results.empty()) {
    return MatchResult{"Unknown", 0.0};
  }
  MatchResult best_match = results[0];
  for (const MatchResult &result : results) {
    if (result.score > best_match.score) {
      best_match = result;
    }
  }
  return best_match;
}
