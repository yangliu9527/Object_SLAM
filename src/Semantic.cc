#include "Semantic.h"
#include "Tracking.h"
#include <fstream>

namespace ORB_SLAM2
{

Semantic::Semantic()
{
    mVaildObjLabelsTUMRGBD = {0,39,41,56,58,62,63,64,65,66,73,77};
    mVaildObjLabelsKitti = {2};
}

void Semantic::ReadSemanticKittiStereo(vector<semantic> &semantics,string &path, double timestamp, float SemanticProbThreshold)
{
    stringstream filenametream;
    filenametream <<  setw(6) << setfill('0')<<to_string(kitti_id);
    string filename;
    filenametream >> filename;

    string semanticpath = path+filename;
    ifstream semanticfile((semanticpath+"/"+filename+".txt").c_str());
    int k =0;
    while(!semanticfile.eof())
    {
        string line;
        getline(semanticfile, line);
        if(!line.empty())
        {
            stringstream ss;
            ss << line;
            int label, x, y, w, h, instance_id;
            float prob;
            ss >> label;
            ss >> prob;
            if(prob <= SemanticProbThreshold)
                continue;
            ss >> x;
            ss >> y;
            ss >> w;
            ss >> h;
            ss >> instance_id;
            if(count(mVaildObjLabelsKitti.begin(),mVaildObjLabelsKitti.end(),label))
            {
                cv::Mat mask = cv::imread(semanticpath+"/"+to_string(instance_id)+".png",-1);
                semantic semantic_temp = {label,prob,x,y,w,h,mask};
                semantics.push_back(semantic_temp);
            }
            
        }

    }
    kitti_id++;

}

void Semantic::ReadSemanticTUMRGBD(vector<semantic> &semantics,string &path, double timestamp, float SemanticProbThreshold)
{

    string filename = to_string(timestamp);;
    string semanticpath = path+filename;
    ifstream semanticfile((semanticpath+"/"+filename+".txt").c_str());
    while(!semanticfile.eof())
    {
        string line;
        getline(semanticfile, line);
        if(!line.empty())
        {
            stringstream ss;
            ss << line;
            int label, x, y, w, h, instance_id;
            float prob;
            ss >> label;
            if(label == 63)
            {
                label = 62;
            }
            ss >> prob;
            if(prob <= SemanticProbThreshold)
                continue;
            ss >> x;
            ss >> y;
            ss >> w;
            ss >> h;
            ss >> instance_id;
            if(count(mVaildObjLabelsTUMRGBD.begin(),mVaildObjLabelsTUMRGBD.end(),label))
            {
                cv::Mat mask = cv::imread(semanticpath+"/"+to_string(instance_id)+".png",-1);
                semantic semantic_temp = {label,prob,x,y,w,h,mask};
                semantics.push_back(semantic_temp);
            }
            
        }

    }
}

}

