#pragma once
#include <atomic>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>

using namespace cv;
using namespace std;

enum dbLoadStatus { NOT_LOADED, LOADING, LOADED };

inline string getLoadStatusString(dbLoadStatus status) {
  switch (status) {
  case NOT_LOADED:
    return "NOT_LOADED";
  case LOADING:
    return "LOADING";
  case LOADED:
    return "LOADED";
  default:
    return "UNKNOWN";
  }
}

/**
 * Structure to hold one match result.
 */
struct MatchResult {
  string name;
  float score;
};

/**
 * Structure to hold all match results.
 */
struct MatchResults {
  vector<MatchResult> results;
  string bestmatch;
};

class DetectedFace {
public:
  string name = "Unknown";
  Mat facedetect;
  Mat feature;
  Size originalSize;

  DetectedFace(const string &n, const Mat &fdetect, const Mat &feat = Mat(),
               const Size &original_size = Size())
      : name(n), facedetect(fdetect), feature(feat), originalSize(original_size) {}

  Rect2i bbox() const {
    if (facedetect.empty())
      return Rect2i();
    int x = static_cast<int>(facedetect.at<float>(0, 0));
    int y = static_cast<int>(facedetect.at<float>(0, 1));
    int w = static_cast<int>(facedetect.at<float>(0, 2));
    int h = static_cast<int>(facedetect.at<float>(0, 3));
    return Rect2i(x, y, w, h);
  }
};

/**
 * @class FaceRecognition
 * @brief Handles face recognition, directory hashing, and feature storage.
 */
class FaceRecognition {
public:
  /**
   * Initializes the detection and recognition models.
   */
  FaceRecognition(const std::string &fdModelPath = "./models/face_detection_yunet_2023mar.onnx",
                  const std::string &frModelPath = "./models/face_recognition_sface_2021dec.onnx");

  /**
   * Destructor - stops watching thread if running.
   */
  ~FaceRecognition();

  /**
   * Loads the persons database from the specified folder.
   *
   * @param persondb_folder The folder containing the persons database.
   * @param force If true, forces reloading the database even if it's already loaded.
   * @param visualize If true, visualizes the detected faces, writes images with suffix
   *    _visualize in the same folder.
   */
  void loadPersonsDB(filesystem::path persondb_folder, bool force = false, bool visualize = false);

  /**
   * Starts watching the database folder for changes.
   * @param check_interval_seconds How often to check for changes (default: 5 seconds)
   */
  void startWatching(int check_interval_seconds = 5);

  /**
   * Stops watching the database folder.
   */
  void stopWatching();

  /**
   * Performs face recognition on the given frame.
   *
   * @param frame The input frame where faces will be detected and recognized.
   * @return 0 on success, 1 on failure.
   */
  int run(Mat frame);

  /**
   * @brief Annotate the frame with the name of the person
   * @param frame Image to be altered
   * @param face Output of the face detector, containing the bounding box and landmarks
   */
  void annotate_with_name(cv::Mat &frame, const DetectedFace &face);

  // Getter and setter for database path
  filesystem::path getDbPath() const { return dbPath; }
  void setDbPath(const filesystem::path &path) {
    this->isDBLoaded = NOT_LOADED;
    dbPath = path;
  }

private:
  /// @brief Indicates whether the database is loaded.
  atomic<dbLoadStatus> isDBLoaded = NOT_LOADED;

  /// @brief a mapping of name to a vector of features.
  unordered_map<string, vector<Mat>> featuresMap;

  /// @brief Face detection model
  Ptr<FaceDetectorYN> detector;

  /// @brief Face recognition model
  Ptr<FaceRecognizerSF> face_recognizer;

  /* START Stuff for watching the folder */
  /// @brief Database folder path
  filesystem::path dbPath;
  /// @brief Last modification time of the database folder
  filesystem::file_time_type lastModTime;
  /// @brief Watcher thread
  thread watcherThread;
  /// @brief Flag to control watcher thread
  atomic<bool> watcherRunning{false};
  /// @brief Check interval in seconds
  int checkInterval = 5;
  /* END Stuff for watching the folder */

  /**
   * Thread function that monitors the database folder for changes.
   */
  void watcherThreadFunc();

  /**
   * Gets the latest modification time of all files in the database folder.
   */
  filesystem::file_time_type getLatestModTime(const filesystem::path &path);

  /**
   * Visualizes detected faces on the input image.
   *
   * @param input The input image where faces will be visualized.
   * @param frame The frame number for logging purposes.
   * @param faces A matrix containing face detection results.
   * @param thickness The thickness of the bounding box and landmark circles.
   */
  static void visualize(Mat &input, int frame, Mat &faces, int thickness = 2);

  /**
   * Resizes the input frame to a maximum size while maintaining the aspect ratio.
   *
   * @param frame The input frame to be resized.
   * @param maxSize The maximum size for either width or height.
   * @param keepAspectRatio Whether to keep the aspect ratio.
   */
  static void resizeFrame(Mat &frame, int maxSize = 400, bool keepAspectRatio = false);

  /**
   * Extracts features from detected faces in the given frame.
   *
   * @param frame The input frame containing faces.
   * @return A vector of feature matrices for each detected face.
   */
  vector<DetectedFace> extractFeatures(Mat &frame);

  /**
   * Finds the best matching person for the given face feature.
   *
   * @param faceFeature The feature vector of the face to match.
   * @param threshold The similarity threshold for matching.
   * @return A MatchResults object, containing a vector of MatchResult structures
   *  and the name of the best matching person.
   */
  MatchResults findBestMatch(const Mat &faceFeature, float threshold = 0.3f);
};
