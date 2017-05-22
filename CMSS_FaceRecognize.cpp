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

bool check_image_and_convert_to_gray(const cv::Mat& src, cv::Mat& img_gray)
{
  if(src.empty())
    return false;
  if(!src.data)
    return false;

  if (src.channels() == 3)
    cv::cvtColor(src, img_gray, CV_BGR2GRAY);
  else if (src.channels() == 1)
    img_gray = src;
  else {
    std::cout << "Please input color or gray image.\n";
    return false;
  }
  return true;
}

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
  cv::Mat img_gray;
  if (!check_image_and_convert_to_gray(src, img_gray))
    return -1;

  if (faceParam.pyramid_scale < 0 || faceParam.pyramid_scale > 1)
    return -2;
  
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

int CMSS_FA_GetFacePointLocation(const cv::Mat& src,
				 FDPARAM& faceParam, 
				 FAPARAM& facePointParam, 
				 FARESULT& facePointInfo)
{
  cv::Mat img_gray;
  if (!check_image_and_convert_to_gray(src, img_gray))
    return -1;

  if (!is_exist(faceParam.modelpath)) {
    std::cout<<"Detection model: " << faceParam.modelpath << " not exist.\n";
    return -3;
  }

  seeta::ImageData img_data_gray(img_gray.cols, img_gray.rows, img_gray.channels());
  img_data_gray.data = img_gray.data;

  // detection
  FDRESULT face_det;
  CMSS_FD_GetFaceResult(src, faceParam, face_det);
  if (face_det.size() == 0) {
    return 0;
  }
  // alignment
  seeta::FaceAlignment point_detector(facePointParam.modelpath.c_str());
  for (int i = 0; i < face_det.size(); ++i) {
    seeta::FaceInfo face_info;
    face_info.bbox.x = face_det[i].faceRect.x;
    face_info.bbox.y = face_det[i].faceRect.y;
    face_info.bbox.width = face_det[i].faceRect.width;
    face_info.bbox.height = face_det[i].faceRect.height;
    face_info.roll = face_det[i].roll;
    face_info.pitch = face_det[i].pitch;
    face_info.yaw = face_det[i].yaw;
    face_info.score = face_det[i].score;
    
    seeta::FacialLandmark gallery_points[5];
    point_detector.PointDetectLandmarks(img_data_gray, face_info, gallery_points);
    FAPIONTLOC face_and_face_point_loc;
    face_and_face_point_loc.faceLocation = face_det[i];
    face_and_face_point_loc.facePointNum = 5;
    for (int j = 0; j < 5; ++j) {
      cv::Point point(gallery_points[j].x, gallery_points[j].y);
      face_and_face_point_loc.facePointLocation.push_back(point);
    }
    facePointInfo.push_back(face_and_face_point_loc);
  }
  return 1;
}

int CMSS_FR_GetCropFaceandExtraFeature(const cv::Mat& src, 
				       FDPARAM& faceParam, 
				       FAPARAM& facePointParam, 
				       FEPARAM& faceExtrafeaParam, 
				       CROPRESULT& cropface,
				       FARESULT& facePointInfo,
				       FEARESULT&  faceFea)
{
  int ret = CMSS_FA_GetFacePointLocation(src, faceParam, facePointParam, facePointInfo);
  if (ret == -1 || ret == -2 || ret == -3 || ret == 0) {
    return ret;
  }
  if (src.channels() != 3)
    return -1;
  seeta::FaceIdentification face_recognizer(faceExtrafeaParam.modelpath.c_str());
  int crop_width = face_recognizer.crop_width();
  int crop_height = face_recognizer.crop_height();
  int channels = 3;
  int feature_size = face_recognizer.feature_size();
  seeta::ImageData img_data(src.cols, src.rows, src.channels());
  img_data.data = src.data;
  for (int i = 0; i < facePointInfo.size(); ++i) {
    seeta::FacialLandmark llpoint[5];
    seeta::ImageData crop_image(crop_width, crop_height, channels);
    cv::Mat cv_crop_image(crop_height, crop_width, CV_8UC3);
    crop_image.data = cv_crop_image.data;
    for (int j = 0; j < 5; ++j) {
      llpoint[j].x = facePointInfo[i].facePointLocation[j].x;
      llpoint[j].y = facePointInfo[i].facePointLocation[j].y;
    }
    face_recognizer.CropFace(img_data, llpoint, crop_image);
    cropface.push_back(cv_crop_image);
  }
  
  for (int i = 0; i < cropface.size(); ++i) {
    std::vector<float> feature (feature_size);
    seeta::ImageData crop_image(crop_width, crop_height, channels);
    crop_image.data = cropface[i].data;
    face_recognizer.ExtractFeature(crop_image, feature.data()); // c++11
    faceFea.push_back(feature);
  }
  return 1;
}

int CMSS_FR_GetFaceFeatureInfo(FEPARAM& faceExtrafeaParam,
			       FACEFEAINFO& featureInfo )
{
  if (!is_exist(faceExtrafeaParam.modelpath)) {
    std::cout<<"Recognition model: " << faceExtrafeaParam.modelpath << " not exist.\n";
    return -3;
  }
  seeta::FaceIdentification face_recognizer(faceExtrafeaParam.modelpath.c_str());
  featureInfo.featureSize = face_recognizer.feature_size();
  featureInfo.cropfaceWidth = face_recognizer.crop_width();
  featureInfo.cropfaceHeight = face_recognizer.crop_height();
  return 0;
}

float CMSS_FR_CalcSimilarity (float* fea1,
			      float* fea2,
			      int len)
{
  if (len < 1) {
    std::cout << "Please provide feature size\n";
    return -1;
  }
  return simd_dot(fea1, fea2, len)
	  / (sqrt(simd_dot(fea1, fea1, len))
	  * sqrt(simd_dot(fea2, fea2, len)));
}
