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




  /*����������*/
  typedef struct facedetectionParam
  {
//    int min_img_height;
//    int min_img_width;
    float adjust_threshold;	/* Only objects with a detection_confidence > adjust_threshold are output */
    /* int slide_wnd_step_x;   /\*��������x����, ͨ����Ϊ 4 *\/ */
    /* int slide_wnd_step_y;   /\*��������y����, ͨ����Ϊ 4 *\/ */
		
    /* double pyramid_scale;      /\*��������С�߶�, ȡֵ��Χ 0~1*\/ */
    /* double score_thresh;       /\*�÷���ֵ, ����ȡֵ 0.95, 2, 4*\/ */
    float nms;
    const char* modelpath;        /*�������ģ��·��*/
    const char* cfg_path;
  }FDPARAM;


  /*�����ؼ��㶨λ����*/
  typedef struct facePointParam
  {	
    std::string modelpath;        /*������λģ��·��*/
  }FAPARAM;

  /*����������ȡ����*/
  typedef struct faceExtraFeaParam
  {	
    std::string modelpath;        /*������λģ��·��*/
  }FEPARAM;

  /*����λ����Ϣ*/
  typedef struct faceLocation
  {
    cv::Rect faceRect;  /*����λ��*/

    double score;   /*�÷�*/

    /*������̬����*/
    double roll;        /*ת��,��z��*/
    double pitch;       /*������,��x��*/
    double yaw;         /*ƫ��,��y��*/
		
  }FACELOC;


  /*�������ؼ���λ����Ϣ*/
  typedef struct faceandfacepointLocation
  {
    FACELOC faceLocation;               /*����λ��*/
    int facePointNum;                         /*�����ؼ������*/
    std::vector< cv::Point > facePointLocation; /*�����ؼ���λ��*/
  }FAPIONTLOC;

  /*����������Ϣ*/
  typedef struct faceFeatureInfo
  {
    int featureSize;                         /*����ά��*/
    int cropfaceWidth;                       /*����ü����������*/
    int cropfaceHeight;                       /*����ü��������߶�*/
  }FACEFEAINFO;

  /*����λ�ý��*/
  typedef std::vector<  FACELOC >  FDRESULT;

  /*�������ؼ���λ�ý��*/
  typedef std::vector<  FAPIONTLOC >  FARESULT;

  /*��������*/
  typedef std::vector< std::vector< float > >  FEARESULT;

  /*cropface���*/
  typedef std::vector< cv::Mat >  CROPRESULT;

  /*  
   * ������⺯��
   * in : src
   * in : faceParam ����������
   * out : faceInfo ����λ�������Ϣ
   * return :  -1 ����ͼƬ�쳣������ͼƬ�����ݡ�ͨ�����Ե�
   *			-2 ���ò����쳣�������������ͳ�������������Χ��
   *			-3 ģ���쳣������ģ��δ�ҵ���ģ����
   *			 0 ���������������
   *			 1 �����������������
   */
  int CMSS_FD_GetFaceResult(	const cv::Mat& src, 
				FDPARAM& faceParam, 
				FDRESULT& faceInfo);

  /*  
   * ������⼰�ؼ��㶨λ����
   * in : src
   * in : faceParam ����������
   * in : facePointParam �ؼ��㶨λ����
   * out : faceInfo �������ؼ���λ����Ϣ
   * return :  -1 ����ͼƬ�쳣������ͼƬ�����ݡ�ͨ�����Ե�
   *			-2 ���ò����쳣�������������ͳ�������������Χ��
   *			-3 ģ���쳣������ģ��δ�ҵ���ģ����
   *			 0 ���������������
   *			 1 �����������������
   */
  int CMSS_FA_GetFacePointLocation(	const cv::Mat& src,
					FDPARAM& faceParam, 
					FAPARAM& facePointParam, 
					FARESULT& facePointInfo);

  /*  
   * ��������ͼƬ��������ȡ����
   * in : src
   * in : faceParam ����������
   * in : facePointParam �ؼ��㶨λ����
   * in : faceExtrafeaParam ������ȡ����
   * out :cropface ��������ü�ͼƬ	
   * out : facePointInfo �������ؼ���λ����Ϣ
   * out ��faceFea ��������
   * return :  -1 ����ͼƬ�쳣������ͼƬ�����ݡ�ͨ�����Ե�
   *			-2 ���ò����쳣�������������ͳ�������������Χ��
   *			-3 ģ���쳣������ģ��δ�ҵ���ģ����
   *			 0 ���������������
   *			 1 �����������������
   */
  int CMSS_FR_GetCropFaceandExtraFeature(	const cv::Mat& src, 
						FDPARAM& faceParam, 
						FAPARAM& facePointParam, 
						FEPARAM& faceExtrafeaParam, 
						CROPRESULT& cropface,
						FARESULT& facePointInfo,
						FEARESULT&  faceFea);

  /*  
   * ����������Ϣ��ȡ
   * in : faceExtrafeaParam ������ȡ����
   * out :featureInfo ������Ϣ
   * out ��fea ��������
   * return :  
   *			-2 ���ò����쳣�������������ͳ�������������Χ��
   *			-3 ģ���쳣������ģ��δ�ҵ���ģ����
   *			 0 ����
   */
  int CMSS_FR_GetFaceFeatureInfo(	 FEPARAM& faceExtrafeaParam,
					 FACEFEAINFO& featureInfo );

  /*  
   * �����������ƶȼ���
   * in ��fea1 Ҫ�Ƚϵ���������1
   * in ��fea2 Ҫ�Ƚϵ���������2
   * in : len ����ά��
   * return :  
   *			float ���ƶ�
   */
  float CMSS_FR_CalcSimilarity (	float* fea1,
					float* fea2,
					int len);




#ifdef _cplusplus
}
#endif


#endif
