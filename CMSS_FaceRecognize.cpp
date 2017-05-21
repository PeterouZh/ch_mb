#include <opencv2/opencv.hpp>

#include "./CMSS_FaceRecognize.h"
#include "FaceIdentification/include/face_identification.h"
#include "FaceIdentification/include/recognizer.h"
#include "FaceIdentification/include/math_functions.h"

#include "FaceDetection/include/face_detection.h"
#include "FaceAlignment/include/face_alignment.h"


#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <fstream>

bool is_exist(std::string file)
{
  std::fstream _file;
  _file.open(file.c_str(),std::ios::in);
  if(!_file)
  {
    return false;
  }
  return true;
}

int CMSS_FD_GetFaceResult(const cv::Mat& src,
			  FDPARAM& faceParam,
			  FDRESULT& faceInfo)
{
  if(src.empty())
    return -1;
  if(!src.data)
    return -1;
  cv::Mat img_gray;
  if (src.channels() == 3)
    cv::cvtColor(src, img_gray, CV_BGR2GRAY);
  else if (src.channels() == 1)
    img_gray = src;
  else {
    std::cout << "Please input color or gray image.\n";
    return -1;
  }

  if (!is_exist(faceParam.modelpath)) {
    std::cout<<"Detection model: " << faceParam.modelpath << " not exist.\n";
    return -3;
  }
  seeta::FaceDetection detector(faceParam.modelpath);
  detector.SetMinFaceSize(faceParam.min_face_size);
  detector.SetMaxFaceSize(faceParam.max_face_size);
  detector.SetWindowStep(faceParam.slide_wnd_step_x, faceParam.slide_wnd_step_y);
  detector.SetImagePyramidScaleFactor(faceParam.pyramid_scale);
  detector.SetScoreThresh(faceParam.score_thresh);

  seeta::ImageData img_data_gray(img_gray.cols, img_gray.rows, img_gray.channels());
  img_data_gray.data = img_gray.data;
  std::vector<seeta::FaceInfo> gallery_faces = detector.Detect(img_data_gray);
  if (gallery_faces.size() == 0) {
    return 0;
  } else {
    FACELOC temp_loc;
    for (int i = 0; i < gallery_faces.size(); ++i) {
      temp_loc.faceRect.x = gallery_faces[i].bbox.x;
      temp_loc.faceRect.y = gallery_faces[i].bbox.y;
      temp_loc.faceRect.width = gallery_faces[i].bbox.width;
      temp_loc.faceRect.height = gallery_faces[i].bbox.height;
      temp_loc.roll = gallery_faces[i].roll;
      temp_loc.pitch = gallery_faces[i].pitch;
      temp_loc.yaw = gallery_faces[i].yaw;
      temp_loc.score = gallery_faces[i].score;
      faceInfo.push_back(temp_loc);
    }
    return 1;
  }
}
