#ifndef OBJECTTYPES_H
#define OBJECTTYPES_H
#include <string>
#include <iostream>
#include <set>
#include <opencv2/core/core.hpp>
#include "Frame.h"
#include "KeyFrame.h"
#include "Eigen/Core"
#include <map>
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "MapPoint.h"
using namespace std;

namespace ORB_SLAM2
{
    
#define MIN_OBJ3DMP_NUM 5

template <typename T> void ReduceVector(vector<T> &vec, vector<bool> status);
template <typename T> void ReduceVector(vector<T> &vec, vector<bool> status)
{
    int n = status.size();
    int k = 0;
    for(int i=0;i<n;i++)
    {
        if(status[i])
        {
            vec[k] = vec[i];
            k++;
        }
    }
    vec.resize(k);
}

class semantic;
class Frame;
class KeyFrame;
class MapPoint;


//2D Object features in image plane
class Object2D
{
public:
    Object2D(semantic sem, vector<cv::KeyPoint> kps, vector<cv::Mat> desps, vector<float> depths, vector<int> framekp_indices, cv::Mat &HSV_histogram, ORBVocabulary* pORBvocabulary);
    int label;
    float prob;
    float x,y,w,h;
    cv::Mat mask;
    //For optimization
    cv::Mat mDistTransImg;
    //ORB key points in object mask
    vector<cv::KeyPoint> mvKeyPoints;
    int N_Kp;
    vector<MapPoint*> mvpMapPoints;
    int N_Matched_Mp;
    vector<int> mvFrameKpIndices;
    
    //ORB descriptors 
    vector<cv::Mat> mvDescriptors;
    //ORB depth
    vector<float> mvDepths;
    //BoW
    void ComputeBoW(ORBVocabulary* pORBvocabulary);
    //ORBVocabulary* mpORBvocabulary;
    DBoW2::BowVector mBowVec;
    DBoW2::FeatureVector mFeatVec;

    //HSV histogram
    cv::Mat mHSVHistogram;

    //track_id
    int track_id = -1;
    
    static vector<int> DynaLabels;
};

struct ObjectObservation
{
    Eigen::Vector3f ObjectCenterW;
    cv::Mat ObsCamPoseTcw;
    cv::Mat AppearanceVec;
};


class Object3D
{
public:
    Object3D(vector<MapPoint*> vpObj3DMps, int Obj2DidxF, Frame *F);
    Object3D(Object2D &Obj2D, Frame *F);
    Object3D(Object2D &Obj2D, KeyFrame *KF);


    
    int mLabel;
    int mTrackID;
    bool mbVaild;
    //Next track ID
    static long unsigned int nNextId;
    //Center of map point cloud, different from position
    Eigen::Vector3f mCenterW;
    //Bounding Box information
    float mMaxXw,mMaxYw,mMinXw,mMinYw,mMinZw,mMaxZw;
    //Map points in Object3D
    vector<MapPoint*> mvpMapPoints;
    //Observation history
    vector<ObjectObservation> mvObservations;
    //Update with a new Object2D Observation
    void Update(int Obj2DidxF, Frame *F);
    int mUpdateCnt;
    //Update with merge Object3D
    bool mbReplaced;
    Object3D* mpReplaced;

    //Calculate metric information
    void CalculateCenterW();
    void CalculateSize();
    void CalculateCenterAndSize();

    //Reject Outliers
    int mToLastRejectOutliers;
    int mnNewAddedMpNum;
    bool isMpBad(MapPoint* pMp);
    void RejectOutliers();
    void RejectOutliersTEST0(vector<MapPoint*> &vpMps);//weighted standard deviation of clusters//
    void RejectOutliersTEST1(vector<MapPoint*> &vpMps);//weighted standard deviation of clusters && ratio of clusters//
    void RejectOutliersTEST2(vector<MapPoint*> &vpMps);//weighted standard deviation of clusters && standard deviation of points//
    void RejectOutliersTEST3(vector<MapPoint*> &vpMps);//ratio of clusters || standard deviation of points//

    //Reject Outliers By clustering and standard deviation
    void RejectOutliersTEST4(vector<MapPoint*> &vpMps);//--------------ratio of clusters-------------//
    void RejectOutliersTEST5(vector<MapPoint*> &vpMps);//---------standard deviation of points----------//
    void RejectOutliersTEST6(vector<MapPoint*> &vpMps);//------------label count--------------//
    void RejectOutliersTEST7(vector<MapPoint*> &vpMps);//-------------ratio of clusters || standard deviation of points-------------//


};



}

#endif