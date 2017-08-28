#include <opencv2/opencv.hpp>

#include <dlib/dnn.h>
#include <dlib/data_io.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>
#include <dlib/timing.h>

#include "./CMSS_FaceRecognize.h"

#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <math.h>

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

namespace fd {
  using namespace dlib;
  // c++11
  template <long num_filters, typename SUBNET> using con5d = con<num_filters,5,5,2,2,SUBNET>;
  template <long num_filters, typename SUBNET> using con5  = con<num_filters,5,5,1,1,SUBNET>;

  template <typename SUBNET> using downsampler  =
    relu<affine<con5d<32, relu<affine<con5d<32, relu<affine<con5d<16,SUBNET>>>>>>>>>;
  template <typename SUBNET> using rcon5  = relu<affine<con5<45,SUBNET>>>;
  using net_type =
    loss_mmod<con<1,9,9,1,1,rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;
  static bool init_flag = false;
}

int CMSS_FD_GetFaceResult(const cv::Mat& src,
			  FDPARAM& faceParam,
			  FDRESULT& faceInfo)
{
  using namespace dlib;
  if (!is_exist(faceParam.modelpath)) {
    std::cout<<"Detection model: " << faceParam.modelpath << " not exist.\n";
    return -3;
  }
  static fd::net_type net;
  if (!fd::init_flag) {
    deserialize(faceParam.modelpath) >> net;
    std::cout << "\tInit detection\n";
    fd::init_flag = true;
  }
  if (!src.data || src.channels() != 3)
    return -1;
  matrix<rgb_pixel> img;
  cv_image<bgr_pixel> img_bgr(src);
  assign_image(img, img_bgr);
  int count = 0;
  while(img.size() < faceParam.min_img_height * faceParam.min_img_width) {
    pyramid_up(img, pyramid_down<2>()); // up scale 2 times
    count += 1;
  }
  float ratio = powf(2, count);
#ifdef TIMING
  using namespace dlib::timing;
  start(1,"Detection");
#endif
  double adjust_threshold = faceParam.adjust_threshold;
  auto dets = net(img, adjust_threshold);
#ifdef TIMING
  stop(1);
  dlib::timing::print();
  dlib::timing::clear();
#endif
  for (auto&& d : dets) {
    FACELOC temp_loc;
    temp_loc.faceRect.x = d.rect.left() / ratio;
    temp_loc.faceRect.y = d.rect.top() / ratio;
    temp_loc.faceRect.width = d.rect.width() / ratio;
    temp_loc.faceRect.height = d.rect.height() / ratio;
    temp_loc.roll = -1;
    temp_loc.pitch = -1;
    temp_loc.yaw = -1;
    temp_loc.score = d.detection_confidence * 100.;
    faceInfo.push_back(temp_loc);
  }
  if (dets.size() > 0)
    return 1;
  else
    return 0;

}

int CMSS_FA_GetFacePointLocation(const cv::Mat& src,
				 FDPARAM& faceParam, 
				 FAPARAM& facePointParam, 
				 FARESULT& facePointInfo)
{
  using namespace dlib;
  if (!is_exist(faceParam.modelpath)) {
    std::cout<<"Detection model: " << faceParam.modelpath << " not exist.\n";
    return -3;
  }
  // detection
  FDRESULT face_det;
  int ret = CMSS_FD_GetFaceResult(src, faceParam, face_det);
  if (face_det.size() <= 0) {
    return ret;
  }
  array2d<rgb_pixel> img;
  cv_image<bgr_pixel> img_bgr(src);
  assign_image(img, img_bgr);
  // alignment
  static shape_predictor sp;
  static bool init_flag = false;
  if (!init_flag) {
    deserialize(facePointParam.modelpath.c_str()) >> sp;
    std::cout << "\tInit landmark\n";
    init_flag = true;
  }
#ifdef TIMING
  using namespace dlib::timing;
  start(2,"Landmark");
#endif
  for (int i = 0; i < face_det.size(); ++i) {
    dlib::rectangle det(face_det[i].faceRect.x, face_det[i].faceRect.y,
			face_det[i].faceRect.x + face_det[i].faceRect.width,
			face_det[i].faceRect.y + face_det[i].faceRect.height);
    full_object_detection shape = sp(img, det);
    FAPIONTLOC face_and_face_point_loc;
    face_and_face_point_loc.faceLocation = face_det[i];
    face_and_face_point_loc.facePointNum = 68;
    for (int j = 0; j < 68; ++j) {
      cv::Point point(shape.part(j).x(), shape.part(j).y());
      face_and_face_point_loc.facePointLocation.push_back(point);
    }
    facePointInfo.push_back(face_and_face_point_loc);
  }
#ifdef TIMING
  stop(2);
  dlib::timing::print();
  dlib::timing::clear();
#endif
  return 1;
}

int CMSS_FR_GetFaceFeatureInfo(FEPARAM& faceExtrafeaParam,
			       FACEFEAINFO& featureInfo )
{
  if (!is_exist(faceExtrafeaParam.modelpath)) {
    std::cout<<"Recognition model: " << faceExtrafeaParam.modelpath << " not exist.\n";
    return -3;
  }
  featureInfo.featureSize = 128;
  featureInfo.cropfaceWidth = 150;
  featureInfo.cropfaceHeight = 150;
  return 0;
}

namespace fr {
  using namespace dlib;
  template <template <int,template<typename>class,int,typename> class block,
	    int N, template<typename>class BN, typename SUBNET>
  using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

  template <template <int,template<typename>class,int,typename> class block,
	    int N, template<typename>class BN, typename SUBNET>
  using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

  template <int N, template <typename> class BN, int stride, typename SUBNET> 
  using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

  template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
  template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

  template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
  template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
  template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
  template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
  template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

  using anet_type =
    loss_metric<fc_no_bias<128,avg_pool_everything<
				 alevel0<
				   alevel1<
				     alevel2<
				       alevel3<
					 alevel4<
					   max_pool<3,3,2,2,
						    relu<
						      affine<
							con<32,7,7,2,2,
							    input_rgb_image_sized<150>
									    >>>>>>>>>>>>;
  
}

int CMSS_FR_GetCropFaceandExtraFeature(const cv::Mat& src, 
				       FDPARAM& faceParam, 
				       FAPARAM& facePointParam, 
				       FEPARAM& faceExtrafeaParam, 
				       CROPRESULT& cropface,
				       FARESULT& facePointInfo,
				       FEARESULT&  faceFea)
{
  using namespace dlib;
  if (facePointInfo.size() <= 0) {
    int ret = CMSS_FA_GetFacePointLocation(src, faceParam, facePointParam, facePointInfo);    
    if (ret <= 0) {
      return ret;
    }
  }

  static fr::anet_type net;
  static bool init_flag = false;
  if (!init_flag) {
    dlib::deserialize(faceExtrafeaParam.modelpath.c_str()) >> net;    
    std::cout << "\tInit recognition\n";
    init_flag = true;
  }
  matrix<rgb_pixel> img;
  cv_image<bgr_pixel> img_bgr(src);
  assign_image(img, img_bgr);
  FACEFEAINFO featureInfo;
  CMSS_FR_GetFaceFeatureInfo(faceExtrafeaParam, featureInfo);
  int crop_size = featureInfo.cropfaceWidth;
  int feat_dim = featureInfo.featureSize;
  std::vector<matrix<rgb_pixel>> faces;
  for (const auto& face : facePointInfo) {
    dlib::rectangle det(face.faceLocation.faceRect.x, face.faceLocation.faceRect.y,
			face.faceLocation.faceRect.x + face.faceLocation.faceRect.width,
			face.faceLocation.faceRect.y + face.faceLocation.faceRect.height);
    std::vector<dlib::point> parts;
    for (const auto& p : face.facePointLocation) {
      parts.push_back(dlib::point(p.x, p.y));
    }
    full_object_detection shape(det, parts);
    matrix<rgb_pixel> face_chip;
    chip_details face_chip_details = get_face_chip_details(shape, crop_size, 0.25);
    extract_image_chip(img, face_chip_details, face_chip);
    faces.push_back(face_chip);
    matrix<bgr_pixel> face_chip_bgr;
    assign_image(face_chip_bgr, face_chip);
    cv::Mat temp = toMat(face_chip_bgr).clone();
    // cv::imshow("test", temp);
    // cv::waitKey();
    cropface.push_back(temp);    
  }
#ifdef TIMING
  using namespace dlib::timing;
  start(3,"Recognition");
#endif
  std::vector<matrix<float,0,1>> face_descriptors = net(faces);
#ifdef TIMING
  stop(3);
  dlib::timing::print();
  dlib::timing::clear();
#endif
  for (const auto& fea : face_descriptors) {
    std::vector<float> feature(fea.begin(), fea.end());
    faceFea.push_back(feature);
  }
  return 1;
}


float CMSS_FR_CalcSimilarity (float* fea1,
			      float* fea2,
			      int len)
{
  if (len < 1) {
    return -1;
  }
  float sum = 0;
  for (int i = 0; i < len; ++i) {
    sum += powf(fea1[i] - fea2[i], 2);
  }
  return sqrtf(sum);
}

int CMSS_FA_GetFacePointLocationGivenFaceLocation(const cv::Mat& src,
						  FDRESULT& faceInfo,
						  FAPARAM& facePointParam, 
						  FARESULT& facePointInfo)
{
  using namespace dlib;
  // detection
  FDRESULT face_det = faceInfo;
  array2d<rgb_pixel> img;
  cv_image<bgr_pixel> img_bgr(src);
  assign_image(img, img_bgr);
  // alignment
  static shape_predictor sp;
  static bool init_flag = false;
  if (!init_flag) {
    deserialize(facePointParam.modelpath.c_str()) >> sp;
    std::cout << "\tInit landmark\n";
    init_flag = true;
  }
#ifdef TIMING
  using namespace dlib::timing;
  start(2,"Landmark");
#endif
  for (int i = 0; i < face_det.size(); ++i) {
    dlib::rectangle det(face_det[i].faceRect.x, face_det[i].faceRect.y,
			face_det[i].faceRect.x + face_det[i].faceRect.width,
			face_det[i].faceRect.y + face_det[i].faceRect.height);
    full_object_detection shape = sp(img, det);
    FAPIONTLOC face_and_face_point_loc;
    face_and_face_point_loc.faceLocation = face_det[i];
    face_and_face_point_loc.facePointNum = 68;
    for (int j = 0; j < 68; ++j) {
      cv::Point point(shape.part(j).x(), shape.part(j).y());
      face_and_face_point_loc.facePointLocation.push_back(point);
    }
    facePointInfo.push_back(face_and_face_point_loc);
  }
#ifdef TIMING
  stop(2);
  dlib::timing::print();
  dlib::timing::clear();
#endif
  return 1;
}
