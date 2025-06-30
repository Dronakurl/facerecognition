#include "facerecognition.hpp"
#include "helper.hpp"
#include <CLI11.hpp>
#include <chrono>
#include <opencv2/imgcodecs.hpp>
#include <thread>

using namespace cv;
using namespace std;

/// @brief Just run the face recognition on one image
int simple(string imagePath, string dbPath) {
  Mat frame = imread(imagePath);
  FaceRecognition facerecognizer;
  facerecognizer.loadPersonsDB(dbPath);
  facerecognizer.run(frame, 0.4, true);
  imwrite("./media/result.jpg", frame);
  return 0;
}

/// @brief Test the folder update mechanism
int test_mode(string imagePath, string dbPath) {

  FR_INFO("=== Face Recognition Async Database Test ===");

  // Initialize face recognition
  FR_INFO("1. Initializing FaceRecognition...");
  FaceRecognition facerecognizer;

  // Load the initial database
  FR_INFO("2. Loading initial persons database from: ");
  facerecognizer.loadPersonsDB(dbPath);

  // Start watching for database changes (check every 2 seconds for faster testing)
  FR_INFO("3. Starting database watcher (check interval: 2 seconds)...");
  facerecognizer.startWatching(2);

  // Load and process the test image
  FR_INFO("4. Loading test image: ");
  Mat frame = imread(imagePath);
  if (frame.empty()) {
    cerr << "Error: Could not load image " << imagePath << endl;
    return 1;
  }

  FR_INFO("5. Running face recognition on test image...");
  MatchResult result = facerecognizer.run_one_face(frame);
  FR_INFO("Found name: %s", result.toString().c_str());

  // Wait a bit to let any initial processing complete
  FR_INFO("6. Waiting 3 seconds...");
  this_thread::sleep_for(chrono::seconds(3));

  // Now trigger a database change by creating an empty JPG file
  FR_INFO("7. Triggering database change by creating an empty JPG file...");
  filesystem::path subfolderPath = filesystem::path(dbPath) / "misterx";
  filesystem::path testFilePath = subfolderPath / "testme.jpg";

  // Ensure the subfolder exists
  if (!filesystem::exists(subfolderPath)) {
    filesystem::create_directory(subfolderPath);
  }

  // Create an empty JPG file
  Mat whiteImage(400, 400, CV_8UC3, Scalar(255, 255, 255));
  imwrite(testFilePath.string(), whiteImage);
  if (filesystem::exists(testFilePath)) {
    FR_INFO("   Created test file: ");
  } else {
    FR_WARNING("   Error: Could not create test file");
  }

  // Wait for the watcher to detect the change and reload
  FR_INFO("8. Waiting 10 seconds for database watcher to detect change...");
  FR_INFO("   (Watch the debug output for 'Database folder changed, reloading...')");
  this_thread::sleep_for(chrono::seconds(10));

  // Run face recognition again to show it's still working
  FR_INFO("9. Running face recognition again after database reload...");
  Mat frame2 = imread(imagePath);
  result = facerecognizer.run_one_face(frame2);
  FR_DEBUG("Found name: %s", result.name.c_str());

  // Clean up the test file
  FR_INFO("10. Cleaning up test file...");
  try {
    if (filesystem::exists(testFilePath)) {
      filesystem::remove(testFilePath);
      FR_INFO("    Test file removed.");
    }
  } catch (const exception &e) {
    FR_ERROR("    Error removing test file: ");
  }

  // Stop watching (this will be done automatically by destructor, but let's be explicit)
  FR_INFO("11. Stopping database watcher...");
  facerecognizer.stopWatching();

  FR_INFO("=== Test completed ===");
  FR_INFO("Expected behavior:");
  FR_INFO("- Initial database load");
  FR_INFO("- First face recognition run");
  FR_INFO("- Database change detection and automatic reload");
  FR_INFO("- Second face recognition run working normally");

  return 0;
}

int main(int argc, char **argv) {
  disableCoreDumps();

  CLI::App app("Face Recognition CLI Tool");
  string imagePath = "/app/media/testdata/IMG.jpg";
  string dbPath = "/app/media/db";
  bool isTestMode = false;

  app.add_option("-i,--image", imagePath, "Path to the input image")->check(CLI::ExistingFile);
  app.add_option("-d,--db", dbPath, "Path to the faces database")->check(CLI::ExistingDirectory);
  app.add_flag("-t,--test-mode", isTestMode, "Run in mode to test database update");
  CLI11_PARSE(app, argc, argv);

  if (!isTestMode)
    simple(imagePath, dbPath);
  else
    test_mode(imagePath, dbPath);

  return 0;
}
