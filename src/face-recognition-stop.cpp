#include <iostream>
#include <fstream>

using std::fstream;

int main(int argc, char **argv) {
  fstream stopFile;
  stopFile.open("/home/pi/projects/facerecognition/stop-video-capture.txt", std::ios::out);
  stopFile << fflush;
  stopFile.close();
}

