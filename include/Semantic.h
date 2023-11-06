#ifndef SEMANTIC_H
#define SEMANTIC_H
#include <string>
#include <iostream>
#include <set>
#include <opencv2/core/core.hpp>
using namespace std;

namespace ORB_SLAM2
{

struct semantic
{
    int label;
    float prob;
    float x,y,w,h;
    cv::Mat mask;
};

class Semantic{

public:
    //TO DO: extract semantic information online
    Semantic(const string &ParamFile, const string &WeightFile, const string &LabelFile);
    Semantic();
    set<int>mVaildObjLabelsKitti;
    set<int>mVaildObjLabelsTUMRGBD;
    //set<int>mVaildObjLabelsTUMRGBD = {0,2,13,56,58,62,64,73,77,75};
    //set<int>mVaildObjLabelsTUMRGBD = {62};
    //Load semantic information offline
    int kitti_id = 0;
    void ReadSemanticKittiStereo(vector<semantic> &semantics,string &path, double timestamp, float prob_threshold);
    void ReadSemanticTUMRGBD(vector<semantic> &semantics,string &path, double timestamp, float prob_threshold);
};



}

#endif