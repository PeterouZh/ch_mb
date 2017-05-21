#include <iostream>
#include <string>

#include "CMSS_FaceRecognize.h"


int main(int argc, char *argv[])
{
  std::string img1_file = "../SeetaFaceEngine/FaceIdentification/data/test_face_recognizer/images/compare_im/Aaron_Peirsol_0001.jpg";
  // test face detection
  std::string det_model = "../SeetaFaceEngine/FaceIdentification/model/seeta_fd_frontal_v1.0.bin";
  FDPARAM faceParam = {20, 40, 4, 4, 0.8, 2.0, const_cast< char* >(det_model.c_str())};
  cv::Mat img1 = cv::imread(img1_file.c_str());
  FDRESULT faceInfo;
  
  CMSS_FD_GetFaceResult(img1, faceParam, faceInfo);
  if (faceInfo.size() < 1) {
    std::cout << "Detect no face.\n";
  } else {
    cv::Mat img_rect = img1;
    for (int i = 0; i < faceInfo.size(); ++i) {
      cv::rectangle(img_rect, cv::Point(faceInfo[i].faceRect.x, faceInfo[i].faceRect.y),
		    cv::Point(faceInfo[i].faceRect.x + faceInfo[i].faceRect.width,
			      faceInfo[i].faceRect.y + faceInfo[i].faceRect.height),
		    cv::Scalar(255, 0, 0), 3);
    }
    cv::imshow("detection", img_rect);
    cv::waitKey(0);
  }
  return 0;
}
