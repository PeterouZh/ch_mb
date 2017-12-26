#include <opencv2/opencv.hpp>

//extern "C"
//{
#include "darknet.h"
typedef layer darklayer;
//}

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

//template <long num_filters, typename SUBNET> using con5d = con<num_filters,3,3,2,2,SUBNET>;
//template <long num_filters, typename SUBNET> using con5  = con<num_filters,3,3,1,1,SUBNET>;
//
//template <typename SUBNET> using downsampler =
//  relu<affine<con5d<64, relu<affine<con5d<32, relu<affine<con5d<16,SUBNET>>>>>>>>>;
//template <typename SUBNET> using rcon5  = relu<affine<con5<45,SUBNET>>>;
//
//using net_type =
//  loss_mmod<con<1,9,9,1,1,rcon5<rcon5<rcon5<downsampler<input_rgb_image_pyramid<pyramid_down<6>>>>>>>>;

    static bool init_flag = false;
}

void mat_into_image(const cv::Mat& src, image im)
{
  unsigned char *data = (unsigned char *)src.data;
  int h = src.rows;
  int w = src.cols;
  int c = src.channels();
  int step = src.step.buf[0]; // w * 3
  int i, j, k;

  for(i = 0; i < h; ++i){
    for(k= 0; k < c; ++k){
      for(j = 0; j < w; ++j){
        im.data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;
      }
    }
  }
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
  static bool init_flag = false;
  static network *net;
  static darklayer l;
  static box *boxes;
  static float **probs;
  static float **masks;
  if (!init_flag) {
    net = load_network(const_cast<char*>(faceParam.cfg_path), const_cast<char*>(faceParam.modelpath), 0);
    set_batch_network(net, 1);
    std::cout << "\tInit detection\n";

    l = net->layers[net->n-1];
    boxes = (box*) calloc(l.w*l.h*l.n, sizeof(box));
    probs = (float**) calloc(l.w*l.h*l.n, sizeof(float *));
    int j;
    for(j = 0; j < l.w*l.h*l.n; ++j)
      probs[j] = (float*) calloc(l.classes + 1, sizeof(float *));
    masks = 0;
    if (l.coords > 4){
      masks = (float**) calloc(l.w*l.h*l.n, sizeof(float*));
      for(j = 0; j < l.w*l.h*l.n; ++j)
        masks[j] = (float*) calloc(l.coords-4, sizeof(float *));
    }

    init_flag = true;
  }
  if (!src.data || src.channels() != 3)
    return -1;

#ifdef TIMING
  using namespace dlib::timing;
  start(1,"Detection");
#endif

  image img = make_image(src.cols, src.rows, src.channels());
  mat_into_image(src, img);
  rgbgr_image(img);
  image sized = letterbox_image(img, net->w, net->h);
  float *X = sized.data;

  network_predict(net, X);
  get_region_boxes(l, img.w, img.h, net->w, net->h, faceParam.adjust_threshold, probs, boxes, masks, 0, 0, 0.5, 1);
  if (faceParam.nms > 0 && faceParam.nms < 1)
    do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, faceParam.nms);

  int i,j;
  int num = l.w*l.h*l.n;
  for(i = 0; i < num; ++i){
    int class_id = -1;
    float tmp_prob = 0;
    for(j = 0; j < l.classes; ++j){
      if (probs[i][j] > faceParam.adjust_threshold){
        if (class_id < 0 || probs[i][j] > tmp_prob) {
          class_id = j;
          tmp_prob = probs[i][j];
        }
      }
    }
    if(class_id >= 0){
      box b = boxes[i];
      int left  = (b.x-b.w/2.)*img.w;
      int right = (b.x+b.w/2.)*img.w;
      int top   = (b.y-b.h/2.)*img.h;
      int bot   = (b.y+b.h/2.)*img.h;

      if(left < 0) left = 0;
      if(right > img.w-1) right = img.w-1;
      if(top < 0) top = 0;
      if(bot > img.h-1) bot = img.h-1;

      FACELOC temp_loc;
      temp_loc.faceRect.x = left;
      temp_loc.faceRect.y = top;
      temp_loc.faceRect.width = right - left;
      temp_loc.faceRect.height = bot - top;
      temp_loc.roll = -1;
      temp_loc.pitch = -1;
      temp_loc.yaw = -1;
      temp_loc.score = tmp_prob * 100.;
      faceInfo.push_back(temp_loc);

    }
  }
  free_image(img);
  free_image(sized);


#ifdef TIMING
  stop(1);
  dlib::timing::print();
  dlib::timing::clear();
#endif
  if (faceInfo.size() > 0)
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
  dlib::matrix<rgb_pixel> img;
  cv_image<bgr_pixel> img_bgr(src);
  assign_image(img, img_bgr);
  FACEFEAINFO featureInfo;
  CMSS_FR_GetFaceFeatureInfo(faceExtrafeaParam, featureInfo);
  int crop_size = featureInfo.cropfaceWidth;
  int feat_dim = featureInfo.featureSize;
  std::vector<dlib::matrix<rgb_pixel>> faces;
  for (const auto& face : facePointInfo) {
    dlib::rectangle det(face.faceLocation.faceRect.x, face.faceLocation.faceRect.y,
                        face.faceLocation.faceRect.x + face.faceLocation.faceRect.width,
                        face.faceLocation.faceRect.y + face.faceLocation.faceRect.height);
    std::vector<dlib::point> parts;
    for (const auto& p : face.facePointLocation) {
      parts.push_back(dlib::point(p.x, p.y));
    }
    full_object_detection shape(det, parts);
    dlib::matrix<rgb_pixel> face_chip;
    chip_details face_chip_details = get_face_chip_details(shape, crop_size, 0.25);
    extract_image_chip(img, face_chip_details, face_chip);
    faces.push_back(face_chip);
    dlib::matrix<bgr_pixel> face_chip_bgr;
    assign_image(face_chip_bgr, face_chip);
    cv::Mat temp = toMat(face_chip_bgr).clone();
//     cv::imshow("test", temp);
//     cv::waitKey();
    cropface.push_back(temp);
  }
#ifdef TIMING
  using namespace dlib::timing;
  start(3,"Recognition");
#endif
  std::vector<dlib::matrix<float,0,1>> face_descriptors = net(faces);
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
