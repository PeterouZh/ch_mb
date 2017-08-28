#include <stdio.h>
#include <sstream> 
#include <iostream>
#include <string>
#include <fstream>

#include "CMSS_FaceRecognize.h"

/* get face point location using given face rect */
int CMSS_FA_GetFacePointLocationGivenFaceLocation(const cv::Mat& src,
						  FDRESULT& faceInfo,
						  FAPARAM& facePointParam, 
						  FARESULT& facePointInfo);

int main(int argc, char *argv_[])
{
  // if (argc < 1) {
  //   std::cout << "Usage error" << std::endl;
  //   return -1;
  // }
  const char *argv[] = {"program",
			"../lfw/lfw",
			"../lfw/lfw_aligned/",
  			"../lfw/lfw_test_list.txt",
			"../lfw/lfw_features.bin"};
  int arg_pos = 0;
  std::string imgs_dir = argv[++arg_pos];
  std::string imgs_align_dir = argv[++arg_pos];
  std::string imgs_list = argv[++arg_pos];
  std::string features_file = argv[++arg_pos];
  
  std::string det_model = "/home/shhs/usr/soft/dlib/examples/build/face_det/mmod_network.dat.dlib";
  std::string align_model = "/home/shhs/usr/soft/dlib/examples/build/face_det/landmark";
  std::string id_model = "/home/shhs/usr/soft/dlib/examples/build/face_det/metric_network_renset.dat.dlib";

  int min_img_height = 200;
  int min_img_width = 200;
  double adjust_threshold = 0;
  FDPARAM faceParam = {min_img_height, min_img_width,
		       adjust_threshold, const_cast< char* >(det_model.c_str())};
  FEPARAM faceExtrafeaParam = {const_cast<char *>(id_model.c_str())};
  FACEFEAINFO featureInfo;
  CMSS_FR_GetFaceFeatureInfo(faceExtrafeaParam, featureInfo);
  int feature_dim = featureInfo.featureSize;

  std::ofstream o_fea(features_file, std::ios::binary);
  std::ofstream logfile("../lfw/log.txt");
  std::ifstream infile(imgs_list);
  std::string line;
  int count = 0;
  while (std::getline(infile, line))
    {
      std::cout << "processing [" << count << '/' << 13233 << ']' << std::endl;
      count++;
      std::istringstream iss(line);
      std::string img_name;
      if (!(iss >> img_name)) { break; } // error
      std::string img_path(imgs_dir + '/' + img_name);
      std::string img_align_path(imgs_align_dir + '/' + img_name);
      cv::Mat img = cv::imread(img_path.c_str());
      if (!img.data) {
	logfile << "Read image " << img_path << " failed\n";
      }
      // ****************************************
      // test detection
      // ****************************************
      FDRESULT faceInfo;
      // cv::imshow("test", img);
      // cv::waitKey(0);
      CMSS_FD_GetFaceResult(img, faceParam, faceInfo);

      FAPARAM facePointParam = {const_cast< char* >(align_model.c_str())};
      FARESULT facePointInfo;

      if (faceInfo.size() < 1) {
      	std::cout << "Detect no face : " << img_path << std::endl;
      	logfile << count << "\tdetect no face : " << img_path << std::endl;
	img = cv::imread(img_align_path.c_str());
	if (!img.data) {
	  logfile << "Read image " << img_align_path << " failed\n";
	  break;
	}
      	FACELOC tmp;
      	tmp.faceRect.x = 0;
      	tmp.faceRect.y = 0;
      	tmp.faceRect.width = img.cols;
      	tmp.faceRect.height = img.rows;
	faceInfo.push_back(tmp);
	CMSS_FA_GetFacePointLocationGivenFaceLocation(img, faceInfo,
						      facePointParam, facePointInfo);
      } 
 
      CROPRESULT cropface;
      FEARESULT faceFea;
      CMSS_FR_GetCropFaceandExtraFeature(img, faceParam, facePointParam,
					 faceExtrafeaParam, cropface,
					 facePointInfo, faceFea);
      
      if (faceFea[0].size() > 0) {
	o_fea.write((char*)faceFea[0].data(), sizeof(float) * feature_dim);
      }
      else {
	logfile << count << "\tdetect no face : " << img_path << std::endl;
	std::vector<float> zeros(feature_dim, 0);
	o_fea.write((char*)zeros.data(), sizeof(float) * feature_dim);
	// copy to other dir
	// std::stringstream ss;
	// ss << "noface/" << count << ".jpg";
	// cv::imwrite(ss.str(), img);
      }
      // if (count == 10) {
      // 	o_fea.close();
      // 	logfile.close();
      // 	infile.close();
      // }
    }
  o_fea.close();
  logfile.close();
  infile.close();
  return 0;
}
