#ifndef OBJECTMATCHER_H
#define OBJECTMATCHER_H
#include "Frame.h"


namespace ORB_SLAM2{

class Frame;

class ObjectMatcher{
public:
    ObjectMatcher();
    void MatchTwoFrame(Frame &PreF, Frame &CurF);
    void MatchMapToFrame(vector<Object3D*> vpObject3Ds, Frame &F);
    //int MatchByOpticalFlow(Frame &PreF, Frame &CurF);
    void MatchByHSV(vector<Object3D*> &vpObject3Ds, Frame &F);
    int MatchByORBMatch(Frame &PreF, Frame &CurF);
    void RejectOutliersByRANSAC(vector<cv::Point2f> &pre_pts, vector<cv::Point2f> &cur_pts, vector<uchar> &status);
    float CalCosineSimilarity(cv::Mat &Hist1, cv::Mat &Hist2);
    void ReduceVector(vector<cv::Point2f> &vpts, vector<uchar> &status);
    void ReduceVector(vector<pair<int,int>> &matches, vector<uchar> &status);
    int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);
    void ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3);
    float static Compute3DIoU(Object3D *pObj3D1, Object3D *pObj3D2);
    float static Compute3DIoU(vector<float> BBox3D1, vector<float> BBox3D2);//BBox3D = [MaxX, MaxY, MaxZ, MinX, MinY, MinZ]
    float static Compute3DOverlapRatio(Object3D *pObj3D1, Object3D *pObj3D2);
    float static Compute3DOverlapRatio(vector<float> BBox3D1, vector<float> BBox3D2);//BBox3D = [MaxX, MaxY, MaxZ, MinX, MinY, MinZ]

    

    static const int TH_HIGH;
    static const int TH_LOW;
    static const int HISTO_LENGTH;

    float CosSimThreshold = 0.95;




};


}

#endif