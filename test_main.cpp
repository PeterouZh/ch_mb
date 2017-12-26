#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>

#include "CMSS_FaceRecognize.h"

void plot_rect(cv::Mat img_rect, FDRESULT faceInfo)
{
  for (int i = 0; i < faceInfo.size(); ++i) {
    cv::rectangle(img_rect, cv::Point(faceInfo[i].faceRect.x, faceInfo[i].faceRect.y),
		  cv::Point(faceInfo[i].faceRect.x + faceInfo[i].faceRect.width,
			    faceInfo[i].faceRect.y + faceInfo[i].faceRect.height),
		  cv::Scalar(255, 0, 0), 3);
  }
  return;
}

int main(int argc, char *argv[])
{
  
  std::string img1_file = "1.jpg";
  std::string img2_file = "2.jpg";
  if (argc == 2) {
    img1_file = argv[1];
  }
  // distance less than fr_threshold can be considered as the same person
  float fr_threshold = 0.628; 
  std::string det_model = "models/tiny-yolo-v3.backup";
  std::string cfg_model = "models/tiny-yolo-test-v3.cfg";
  // std::string det_model = "/home/shhs/usr/soft/dlib/examples/build/face_det/mmod_network.dat.dlib";
  std::string align_model = "models/landmark";
  std::string id_model = "models/metric_network_renset.dat.dlib";

  cv::Mat img1 = cv::imread(img1_file.c_str());
  cv::Mat img2 = cv::imread(img2_file.c_str());
  // ****************************************
  // test detection
  // ****************************************
//  int min_img_height = 300;
//  int min_img_width = 300;
  float adjust_threshold = 0.3;
  float nms = 0.3;
  FDPARAM faceParam = {adjust_threshold, nms, det_model.c_str(), cfg_model.c_str()};
  FDRESULT faceInfo;
  FDRESULT faceInfo2;
  CMSS_FD_GetFaceResult(img1, faceParam, faceInfo);
  CMSS_FD_GetFaceResult(img2, faceParam, faceInfo2);
  cv::Mat img_rect = img1.clone();
  cv::Mat img_rect2 = img2.clone();
  if (faceInfo.size() < 1 || faceInfo2.size() < 1) {
    std::cout << "Detect no face.\n";
    return 0;
  } else {
    plot_rect(img_rect, faceInfo);
    cv::namedWindow("detection", cv::WINDOW_NORMAL);
    cv::namedWindow("detection2", cv::WINDOW_NORMAL);
    cv::imshow("detection", img_rect);
    plot_rect(img_rect2, faceInfo2);
    cv::imshow("detection2", img_rect2);
    cv::waitKey(0);
//    return 0;
  }
  // ****************************************
  // test alignment
  // ****************************************
  FAPARAM facePointParam = {const_cast< char* >(align_model.c_str())};
  FARESULT facePointInfo;
  FARESULT facePointInfo2;
  CMSS_FA_GetFacePointLocation(img1, faceParam, facePointParam, facePointInfo);
  CMSS_FA_GetFacePointLocation(img2, faceParam, facePointParam, facePointInfo2);

  
  cv::Mat img_point = img1.clone();
  cv::Mat img_point2 = img2.clone();
  for (int j = 0; j < facePointInfo.size(); ++j) {
    for (int i = 0; i<68; i++) {
	cv::circle(img_point, cv::Point(facePointInfo[j].facePointLocation[i].x,
					facePointInfo[j].facePointLocation[i].y),
		   2, CV_RGB(0, 255, 0));
	cv::circle(img_point2, cv::Point(facePointInfo2[j].facePointLocation[i].x,
					facePointInfo2[j].facePointLocation[i].y),
		   2, CV_RGB(0, 255, 0));
    }
  }
  cv::imshow("alignment", img_point);
  cv::imshow("alignment2", img_point2);
  // cv::waitKey(0);


  // ****************************************
  // test crop and extract feature
  // ****************************************
  FEPARAM faceExtrafeaParam = {const_cast<char *>(id_model.c_str())};
  CROPRESULT cropface;
  FEARESULT faceFea;
  CROPRESULT cropface2;
  FEARESULT faceFea2;
  CMSS_FR_GetCropFaceandExtraFeature(img1, faceParam, facePointParam,
				     faceExtrafeaParam, cropface,
				     facePointInfo, faceFea);
  CMSS_FR_GetCropFaceandExtraFeature(img2, faceParam, facePointParam,
				     faceExtrafeaParam, cropface2,
				     facePointInfo2, faceFea2);

  if (cropface.size() < 1 || cropface2.size() < 1) {
    std::cout << "Crop faces failed\n";
    return -1;
  }
  cv::imshow("crop_face", cropface[0]);
  cv::imshow("crop_face2", cropface2[0]);
  // ****************************************
  // get feature info
  // ****************************************
  FACEFEAINFO featureInfo;
  CMSS_FR_GetFaceFeatureInfo(faceExtrafeaParam, featureInfo);
  printf("\nfeature_size[%d], crop_width[%d], crop_height[%d]\n\n",
	 featureInfo.featureSize, featureInfo.cropfaceWidth, featureInfo.cropfaceHeight);
  
  // ****************************************
  // caculate similarity
  // ****************************************
  std::ofstream of_fea("feature.txt");  
  of_fea << img1_file << "\n";
  for (int i = 0; i < featureInfo.featureSize; ++i) {
    of_fea << faceFea[0][i] << "\n";
  }
  of_fea << "\n" << img2_file << "\n";
  for (int i = 0; i < featureInfo.featureSize; ++i) {
    of_fea << faceFea2[0][i] << "\n";
  }
  of_fea << "\n";
  of_fea.close();

  float sim = CMSS_FR_CalcSimilarity(faceFea[0].data(), faceFea2[0].data(),
				     featureInfo.featureSize);
  std::cout << "Similarity = " << sim 
	    << " < "  << fr_threshold << "[same person]" 
	    << std::endl << std::endl;
  cv::waitKey(0);
  return 0;
}
