/* cmss_facerecognize.h
 * biuld on 20170412
 *
 *
 */

#ifndef CMSS_FACERECOGNIZE_H_
#define CMSS_FACERECOGNIZE_H_

#include <vector>
#include <string>

#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/imgproc/imgproc.hpp> 


#ifdef _cplusplus
extern "C"{
#endif




  /*人脸检测参数*/
  typedef struct facedetectionParam
  {
    int min_img_height;
    int min_img_width;
    double adjust_threshold;	/* Only objects with a detection_confidence > adjust_threshold are output */
    /* int slide_wnd_step_x;   /\*滑动窗口x步长, 通常设为 4 *\/ */
    /* int slide_wnd_step_y;   /\*滑动窗口y步长, 通常设为 4 *\/ */
		
    /* double pyramid_scale;      /\*金字塔缩小尺度, 取值范围 0~1*\/ */
    /* double score_thresh;       /\*得分阈值, 典型取值 0.95, 2, 4*\/ */
	
    char* modelpath;        /*人脸检测模型路径*/
  }FDPARAM;


  /*人脸关键点定位参数*/
  typedef struct facePointParam
  {	
    std::string modelpath;        /*人脸定位模型路径*/
  }FAPARAM;

  /*人脸特征提取参数*/
  typedef struct faceExtraFeaParam
  {	
    std::string modelpath;        /*人脸定位模型路径*/
  }FEPARAM;

  /*人脸位置信息*/
  typedef struct faceLocation
  {
    cv::Rect faceRect;  /*人脸位置*/

    double score;   /*得分*/

    /*人脸姿态参数*/
    double roll;        /*转角,绕z轴*/
    double pitch;       /*俯仰角,绕x轴*/
    double yaw;         /*偏角,绕y轴*/
		
  }FACELOC;


  /*人脸及关键点位置信息*/
  typedef struct faceandfacepointLocation
  {
    FACELOC faceLocation;               /*人脸位置*/
    int facePointNum;                         /*人脸关键点个数*/
    std::vector< cv::Point > facePointLocation; /*人脸关键点位置*/
  }FAPIONTLOC;

  /*人脸特征信息*/
  typedef struct faceFeatureInfo
  {
    int featureSize;                         /*特征维数*/
    int cropfaceWidth;                       /*对齐裁剪后人脸宽度*/
    int cropfaceHeight;                       /*对齐裁剪后人脸高度*/
  }FACEFEAINFO;

  /*人脸位置结果*/
  typedef std::vector<  FACELOC >  FDRESULT;

  /*人脸及关键点位置结果*/
  typedef std::vector<  FAPIONTLOC >  FARESULT;

  /*人脸特征*/
  typedef std::vector< std::vector< float > >  FEARESULT;

  /*cropface结果*/
  typedef std::vector< cv::Mat >  CROPRESULT;

  /*  
   * 人脸检测函数
   * in : src
   * in : faceParam 人脸检测参数
   * out : faceInfo 人脸位置相关信息
   * return :  -1 传入图片异常，包括图片无数据、通道不对等
   *			-2 设置参数异常，包括参数类型出错、参数超过范围等
   *			-3 模型异常，包括模型未找到及模型损坏
   *			 0 检测正常，无人脸
   *			 1 检测正常，存在人脸
   */
  int CMSS_FD_GetFaceResult(	const cv::Mat& src, 
				FDPARAM& faceParam, 
				FDRESULT& faceInfo);

  /*  
   * 人脸检测及关键点定位函数
   * in : src
   * in : faceParam 人脸检测参数
   * in : facePointParam 关键点定位参数
   * out : faceInfo 人脸及关键点位置信息
   * return :  -1 传入图片异常，包括图片无数据、通道不对等
   *			-2 设置参数异常，包括参数类型出错、参数超过范围等
   *			-3 模型异常，包括模型未找到及模型损坏
   *			 0 检测正常，无人脸
   *			 1 检测正常，存在人脸
   */
  int CMSS_FA_GetFacePointLocation(	const cv::Mat& src,
					FDPARAM& faceParam, 
					FAPARAM& facePointParam, 
					FARESULT& facePointInfo);

  /*  
   * 人脸对齐图片及特征获取函数
   * in : src
   * in : faceParam 人脸检测参数
   * in : facePointParam 关键点定位参数
   * in : faceExtrafeaParam 特征提取参数
   * out :cropface 人脸对齐裁剪图片	
   * out : facePointInfo 人脸及关键点位置信息
   * out ：faceFea 特征序列
   * return :  -1 传入图片异常，包括图片无数据、通道不对等
   *			-2 设置参数异常，包括参数类型出错、参数超过范围等
   *			-3 模型异常，包括模型未找到及模型损坏
   *			 0 检测正常，无人脸
   *			 1 检测正常，存在人脸
   */
  int CMSS_FR_GetCropFaceandExtraFeature(	const cv::Mat& src, 
						FDPARAM& faceParam, 
						FAPARAM& facePointParam, 
						FEPARAM& faceExtrafeaParam, 
						CROPRESULT& cropface,
						FARESULT& facePointInfo,
						FEARESULT&  faceFea);

  /*  
   * 人脸特征信息获取
   * in : faceExtrafeaParam 特征提取参数
   * out :featureInfo 特征信息
   * out ：fea 特征序列
   * return :  
   *			-2 设置参数异常，包括参数类型出错、参数超过范围等
   *			-3 模型异常，包括模型未找到及模型损坏
   *			 0 正常
   */
  int CMSS_FR_GetFaceFeatureInfo(	 FEPARAM& faceExtrafeaParam,
					 FACEFEAINFO& featureInfo );

  /*  
   * 人脸特征相似度计算
   * in ：fea1 要比较的特征序列1
   * in ：fea2 要比较的特征序列2
   * in : len 特征维数
   * return :  
   *			float 相似度
   */
  float CMSS_FR_CalcSimilarity (	float* fea1,
					float* fea2,
					int len);




#ifdef _cplusplus
}
#endif


#endif
