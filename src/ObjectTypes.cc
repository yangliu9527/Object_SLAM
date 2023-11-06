#include "ObjectTypes.h"
#include "Semantic.h"
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include<chrono>


namespace ORB_SLAM2{

vector<int> Object2D::DynaLabels = {0,1,2,3};
long unsigned int Object3D::nNextId=0;

Object2D::Object2D(semantic sem, vector<cv::KeyPoint> kps, vector<cv::Mat> desps, vector<float> depths, vector<int> framekp_indices, cv::Mat &HSV_histogram, ORBVocabulary* pORBvocabulary):
mask(sem.mask),label(sem.label),prob(sem.prob),x(sem.x),y(sem.y),w(sem.w),h(sem.h),mvKeyPoints(kps),mvDescriptors(desps),mvDepths(depths), mvFrameKpIndices(framekp_indices), mHSVHistogram(HSV_histogram)
{
    mvpMapPoints = vector<MapPoint*>(kps.size(),static_cast<MapPoint*>(NULL));
    ComputeBoW(pORBvocabulary);
    cv::distanceTransform(~mask, mDistTransImg, cv::DIST_L2,cv::DIST_MASK_PRECISE);
}


void Object2D::ComputeBoW(ORBVocabulary* pORBvocabulary)
{
    if(mBowVec.empty())
    {
        pORBvocabulary->transform(mvDescriptors,mBowVec,mFeatVec,4);
    }
}

Object3D::Object3D(vector<MapPoint*> vpObj3DMps, int Obj2DidxF, Frame *F):
mvpMapPoints(vpObj3DMps),mpReplaced(static_cast<Object3D*>(NULL))
{

    Object2D &Obj2D = F->mvObject2Ds[Obj2DidxF];
    mTrackID = nNextId++;
    mLabel = Obj2D.label;
    Obj2D.track_id = mTrackID;
    CalculateCenterAndSize();
    ObjectObservation Obs = { mCenterW, F->mTcw.clone(), Obj2D.mHSVHistogram.clone()};
    mvObservations.push_back(Obs);
    mUpdateCnt = 0;
    mbVaild = true;
    for(auto p: mvpMapPoints)
    {
        p->SetInsertedTime(mTrackID, 0);
    }
    //cout<<"Initialize a new Object3D with "<<vpObj3DMps.size()<<" MapPoints, TrackID= "<<mTrackID<<", Class= "<<mLabel<<endl;
}

void Object3D::Update(int Obj2DidxF, Frame *F)
{
    mUpdateCnt++;
    Object2D &Obj2D = F->mvObject2Ds[Obj2DidxF];
    int label = Obj2D.label;
    float prob = Obj2D.prob;
    vector<bool> vbFrameOutliers = F->mvbOutlier; 
    Obj2D.track_id = mTrackID;
    vector<int> vFrameKpIndices = Obj2D.mvFrameKpIndices;
    vector<MapPoint*> vpCandidateMps;
    for(int i=0;i<vFrameKpIndices.size();i++)
    {
        MapPoint* pMp = F->mvpMapPoints[vFrameKpIndices[i]];
        if(pMp)
        {
            if(pMp->isBad())
                continue;
            
            if(vbFrameOutliers[vFrameKpIndices[i]])
                continue;
            

            if(!count(mvpMapPoints.begin(), mvpMapPoints.end(), pMp))
            {
                //mvpMapPoints.push_back(pMp);
                vpCandidateMps.push_back(pMp);
                pMp->SetInsertedTime(mTrackID, mUpdateCnt);
            }
            else
            {
                pMp->AddTrackIDCnt(mTrackID);
            }
        }
    }
    if(!vpCandidateMps.empty())
    {
        for(auto p: vpCandidateMps)
        {
            p->SetInsertedTime(mTrackID, 0);
            mvpMapPoints.push_back(p);
            mnNewAddedMpNum++;
        }        
    }

    //-----------RejectOutliers every update time----------//
    //RejectOutliersTEST6(mvpMapPoints);

    //----------RejectOutliers every N update times-------//
    // if(mToLastRejectOutliers == 5)
    // {
    //     RejectOutliersTEST2(mvpMapPoints);
    // }
    // else
    // {
    //     mToLastRejectOutliers++;
    // }

    //----------RejectOutliers every N new Added MapPoints-------//
    //if(mnNewAddedMpNum >=100 )
    {
       

    if(mvpMapPoints.size()>3000)
    {
        RejectOutliersTEST5(mvpMapPoints);
    }
    else
    {
        RejectOutliersTEST7(mvpMapPoints);
    }


    }

    
    CalculateCenterAndSize();  
    ObjectObservation Obs = { mCenterW, F->mTcw.clone(), Obj2D.mHSVHistogram.clone()};
    mvObservations.push_back(Obs);

    //decide if the object is vaild
    if(mUpdateCnt >5 && mvpMapPoints.size()<5)
    {
        mbVaild = false;
    }  
}



bool Object3D::isMpBad(MapPoint* pMp)
{
    bool res;
    res = (pMp->GetLabelProb(mLabel)<0.5);
    return res;
}



void Object3D::RejectOutliers()
{
    if(mvpMapPoints.size()>3000)
    {
        RejectOutliersTEST5(mvpMapPoints);
    }
    else
    {
        RejectOutliersTEST7(mvpMapPoints);
    }
}




//--------------weighted standard deviation of clusters-------------//
void Object3D::RejectOutliersTEST0(vector<MapPoint*> &vpMps)
{
    mToLastRejectOutliers = 0;
    mnNewAddedMpNum = 0;
    cout<<"Reject Outliers for Object3D "<<mTrackID<<"..."<<endl;
    int N_Mps = vpMps.size();
    vector<bool> status(N_Mps, true);
    vector<int> vMpindices;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    for(int i=0; i<N_Mps; i++)
    {
        MapPoint* pMp = vpMps[i];
        if(isMpBad(pMp))
        {
            status[i] = false;
            continue;
        }  
        cv::Mat PosW = pMp->GetWorldPos();
        pcl::PointXYZ p;
        p.x = PosW.at<float>(0);
        p.y = PosW.at<float>(1);
        p.z = PosW.at<float>(2);
        cloud->points.push_back(p);
        vMpindices.push_back(i);
    }

    vector<pcl::PointIndices> eceClustersIndices;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ece;
    tree->setInputCloud(cloud);
    ece.setInputCloud(cloud);
    ece.setClusterTolerance(0.1);
    ece.setMinClusterSize(1);
    ece.setMaxClusterSize(9999);
    ece.setSearchMethod(tree);
    ece.extract(eceClustersIndices);

    int cluster_num = eceClustersIndices.size();
    cout<<"Clustering "<<cluster_num<<" clusters, "<<endl;
    float std_dev = 0.0;
    vector<Eigen::Vector3f> ClusterCentroids;
    Eigen::Vector3f mean = Eigen::Vector3f::Zero();
    for(int j=0;j<cluster_num;j++)
    {
        Eigen::Vector3f ClusterCentroid = Eigen::Vector3f::Zero();
        for(auto index:eceClustersIndices[j].indices)
        {
            Eigen::Vector3f P(cloud->points[index].x, cloud->points[index].y, cloud->points[index].z);
            ClusterCentroid = ClusterCentroid+P;
        }
        ClusterCentroid = ClusterCentroid/(1.0*eceClustersIndices[j].indices.size());
        ClusterCentroids.push_back(ClusterCentroid);
    
        mean = mean+ClusterCentroid;
    }
    
    mean = mean/(1.0*cluster_num);
    cout<<"cluster centroids mean = "<<mean<<endl;
    vector<float> distances;
    for(int j=0;j<cluster_num;j++)
    {
        float distance = (ClusterCentroids[j]-mean).norm();
        distances.push_back(distance);
        std_dev = std_dev+distance*distance;
        
    }
    std_dev = sqrt(std_dev/cluster_num);
    cout<<"Centroids of clusters std dev = "<<std_dev<<endl;

    for(int j=0;j<cluster_num;j++)
    {
        float ratio = (1.0*eceClustersIndices[j].indices.size())/(1.0*N_Mps);
        cout << "cluster "<<j<<" has "<<eceClustersIndices[j].indices.size()<<" points, "<<"the ratio = "<<ratio<<", the centroid is ["<<ClusterCentroids[j][0]<<","<<ClusterCentroids[j][1]<<","<<ClusterCentroids[j][2]<<"]"<<", distance to centroid = "<<distances[j]<<endl;
        if(distances[j]>3.0*ratio*std_dev)
        {
            for(auto index: eceClustersIndices[j].indices)
            {
                status[vMpindices[index]] = 0;
                //status[index] = 0;
            }
        }
    }

    ReduceVector(vpMps, status);
}


//--------------weighted standard deviation of clusters && ratio of clusters-------------//
void Object3D::RejectOutliersTEST1(vector<MapPoint*> &vpMps)
{
    mToLastRejectOutliers = 0;
    mnNewAddedMpNum = 0;
    cout<<"Reject Outliers for Object "<<mTrackID<<" ..."<<endl;
    int N_Mps = vpMps.size();
    vector<bool> status(N_Mps, true);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    vector<int> vMpindices;
    for(int i=0; i<N_Mps; i++)
    {
        MapPoint* pMp = vpMps[i];
        if(isMpBad(pMp))
        {
            status[i] = false;
            continue;
        }  
        cv::Mat PosW = pMp->GetWorldPos();
        pcl::PointXYZ p;
        p.x = PosW.at<float>(0);
        p.y = PosW.at<float>(1);
        p.z = PosW.at<float>(2);
        cloud->points.push_back(p);
        vMpindices.push_back(i);
    }

    vector<pcl::PointIndices> eceClustersIndices;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ece;
    tree->setInputCloud(cloud);
    ece.setInputCloud(cloud);
    ece.setClusterTolerance(0.1);
    ece.setMinClusterSize(1);
    ece.setMaxClusterSize(9999);
    ece.setSearchMethod(tree);
    ece.extract(eceClustersIndices);
    int N_candidates = vMpindices.size();
    int cluster_num = eceClustersIndices.size();
    cout<<"Clustering "<<cluster_num<<" clusters, "<<endl;
    float std_dev = 0.0;
    vector<Eigen::Vector3f> ClusterCentroids;
    Eigen::Vector3f mean = Eigen::Vector3f::Zero();
    for(int j=0;j<cluster_num;j++)
    {
        Eigen::Vector3f ClusterCentroid = Eigen::Vector3f::Zero();
        for(auto index:eceClustersIndices[j].indices)
        {
            Eigen::Vector3f P(cloud->points[index].x, cloud->points[index].y, cloud->points[index].z);
            ClusterCentroid = ClusterCentroid+P;
        }
        ClusterCentroid = ClusterCentroid/(1.0*eceClustersIndices[j].indices.size());
        ClusterCentroids.push_back(ClusterCentroid);
    
        mean = mean+ClusterCentroid;
    }
    
    mean = mean/(1.0*cluster_num);
    cout<<"cluster centroids mean = "<<mean<<endl;
    vector<float> distances;
    for(int j=0;j<cluster_num;j++)
    {
        float distance = (ClusterCentroids[j]-mean).norm();
        distances.push_back(distance);
        std_dev = std_dev+distance*distance;
        
    }
    std_dev = sqrt(std_dev/cluster_num);
    cout<<"Centroids of clusters std dev = "<<std_dev<<endl;

    for(int j=0;j<cluster_num;j++)
    {
        float ratio = (1.0*eceClustersIndices[j].indices.size())/(1.0*N_candidates);
        cout << "cluster "<<j<<" has "<<eceClustersIndices[j].indices.size()<<" points, "<<"the ratio = "<<ratio<<", the centroid is ["<<ClusterCentroids[j][0]<<","<<ClusterCentroids[j][1]<<","<<ClusterCentroids[j][2]<<"]"<<", distance to centroid = "<<distances[j]<<endl;
        if(distances[j]>3.0*ratio*std_dev || ratio < 0.1)
        //if(ratio < 0.1)
        {
            for(auto index: eceClustersIndices[j].indices)
            {
                status[vMpindices[index]] = 0;
                //status[index] = 0;
            }
        }
    }

    ReduceVector(vpMps, status);
}


//--------------weighted standard deviation of clusters && standard deviation of points-------------//
void Object3D::RejectOutliersTEST2(vector<MapPoint*> &vpMps)
{
    mToLastRejectOutliers = 0;
    mnNewAddedMpNum = 0;
    cout<<"Reject Outliers for Object3D "<<mTrackID<<"..."<<endl;
    int N_Mps = vpMps.size();
    vector<bool> status(N_Mps, true);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    cv::Mat CandidatesCenter = cv::Mat::zeros(cv::Size(1,3),CV_32F);
    vector<pair<int,cv::Mat>> vMpindicesPos;
    for(int i=0; i<N_Mps; i++)
    {
        MapPoint* pMp = vpMps[i];
        if(isMpBad(pMp))
        {
            status[i] = false;
            continue;
        }  
        cv::Mat PosW = pMp->GetWorldPos();
        CandidatesCenter = CandidatesCenter+PosW;
        pcl::PointXYZ p;
        p.x = PosW.at<float>(0);
        p.y = PosW.at<float>(1);
        p.z = PosW.at<float>(2);
        cloud->points.push_back(p);
        vMpindicesPos.push_back(make_pair(i,PosW));
    }
    int N_candidates = vMpindicesPos.size();

    //standard deviation
    CandidatesCenter = CandidatesCenter/(1.0*N_candidates);
    vector<float> CandidatesDistances;
    float CandidatesStdDev = 0.0;
    for(int i=0;i<N_candidates;i++)
    {
        float distance = cv::norm(CandidatesCenter-vMpindicesPos[i].second);
        CandidatesStdDev = CandidatesStdDev+distance*distance;
        CandidatesDistances.push_back(distance);
    }
    CandidatesStdDev = sqrt(CandidatesStdDev/(1.0*N_candidates));
    

    //DBSCAN
    vector<pcl::PointIndices> eceClustersIndices;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ece;
    tree->setInputCloud(cloud);
    ece.setInputCloud(cloud);
    ece.setClusterTolerance(0.1);
    ece.setMinClusterSize(1);
    ece.setMaxClusterSize(9999);
    ece.setSearchMethod(tree);
    ece.extract(eceClustersIndices);

    int cluster_num = eceClustersIndices.size();
    cout<<"Clustering "<<cluster_num<<" clusters, "<<endl;
    float std_dev = 0.0;
    vector<Eigen::Vector3f> ClusterCentroids;
    Eigen::Vector3f mean = Eigen::Vector3f::Zero();
    for(int j=0;j<cluster_num;j++)
    {
        Eigen::Vector3f ClusterCentroid = Eigen::Vector3f::Zero();
        for(auto index:eceClustersIndices[j].indices)
        {
            Eigen::Vector3f P(cloud->points[index].x, cloud->points[index].y, cloud->points[index].z);
            ClusterCentroid = ClusterCentroid+P;
        }
        ClusterCentroid = ClusterCentroid/(1.0*eceClustersIndices[j].indices.size());
        ClusterCentroids.push_back(ClusterCentroid);
    
        mean = mean+ClusterCentroid;
    }
    
    mean = mean/(1.0*cluster_num);
    cout<<"cluster centroids mean = "<<mean<<endl;
    vector<float> distances;
    for(int j=0;j<cluster_num;j++)
    {
        float distance = (ClusterCentroids[j]-mean).norm();
        distances.push_back(distance);
        std_dev = std_dev+distance*distance;
        
    }
    std_dev = sqrt(std_dev/cluster_num);
    cout<<"Centroids of clusters std dev = "<<std_dev<<endl;

    for(int j=0;j<cluster_num;j++)
    {
        float ratio = (1.0*eceClustersIndices[j].indices.size())/(1.0*N_candidates);
        cout << "cluster "<<j<<" has "<<eceClustersIndices[j].indices.size()<<" points, "<<"the ratio = "<<ratio<<", the centroid is ["<<ClusterCentroids[j][0]<<","<<ClusterCentroids[j][1]<<","<<ClusterCentroids[j][2]<<"]"<<", distance to centroid = "<<distances[j]<<endl;
        if(distances[j]>3.0*ratio*std_dev)
        {
            for(auto index: eceClustersIndices[j].indices)
            {
                if(CandidatesDistances[index] > 3*CandidatesStdDev)
                    status[vMpindicesPos[index].first] = 0;
                //status[index] = 0;
            }
        }
    }

    ReduceVector(vpMps, status);
}


//-------------ratio of clusters && standard deviation of points-------------//
void Object3D::RejectOutliersTEST3(vector<MapPoint*> &vpMps)
{
    mToLastRejectOutliers = 0;
    mnNewAddedMpNum = 0;
    cout<<"Reject Outliers for Object3D "<<mTrackID<<"..."<<endl;
    int N_Mps = vpMps.size();
    vector<bool> status(N_Mps, true);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    cv::Mat CandidatesCenter = cv::Mat::zeros(cv::Size(1,3),CV_32F);
    vector<pair<int,cv::Mat>> vMpindicesPos;
    for(int i=0; i<N_Mps; i++)
    {     
        MapPoint* pMp = vpMps[i];
        if(isMpBad(pMp))
        {
            status[i] = false;
            continue;
        }  
        cv::Mat PosW = pMp->GetWorldPos();
        CandidatesCenter = CandidatesCenter+PosW;
        pcl::PointXYZ p;
        p.x = PosW.at<float>(0);
        p.y = PosW.at<float>(1);
        p.z = PosW.at<float>(2);
        cloud->points.push_back(p);
        vMpindicesPos.push_back(make_pair(i,PosW));
    }
    int N_candidates = vMpindicesPos.size();

    //standard deviation
    CandidatesCenter = CandidatesCenter/(1.0*N_candidates);
    vector<float> CandidatesDistances;
    float CandidatesStdDev = 0.0;
    for(int i=0;i<N_candidates;i++)
    {
        float distance = cv::norm(CandidatesCenter-vMpindicesPos[i].second);
        CandidatesStdDev = CandidatesStdDev+distance*distance;
        CandidatesDistances.push_back(distance);
    }
    CandidatesStdDev = sqrt(CandidatesStdDev/(1.0*N_candidates));
    

    //DBSCAN
    vector<pcl::PointIndices> eceClustersIndices;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ece;
    tree->setInputCloud(cloud);
    ece.setInputCloud(cloud);
    ece.setClusterTolerance(0.1);
    ece.setMinClusterSize(1);
    ece.setMaxClusterSize(9999);
    ece.setSearchMethod(tree);
    ece.extract(eceClustersIndices);

    int cluster_num = eceClustersIndices.size();
    

    for(int j=0;j<cluster_num;j++)
    {
        float ratio = (1.0*eceClustersIndices[j].indices.size())/(1.0*N_candidates);
        //cout << "cluster "<<j<<" has "<<eceClustersIndices[j].indices.size()<<" points, "<<"the ratio = "<<ratio<<", the centroid is ["<<ClusterCentroids[j][0]<<","<<ClusterCentroids[j][1]<<","<<ClusterCentroids[j][2]<<"]"<<", distance to centroid = "<<distances[j]<<endl;
        if(ratio < 0.1)
        {
            for(auto index: eceClustersIndices[j].indices)
            {
                if(CandidatesDistances[index] > 3*CandidatesStdDev)
                    status[vMpindicesPos[index].first] = 0;
                //status[index] = 0;
            }
        }
    }

    ReduceVector(vpMps, status);
}


//--------------ratio of clusters-------------//
void Object3D::RejectOutliersTEST4(vector<MapPoint*> &vpMps)
{
    mToLastRejectOutliers = 0;
    mnNewAddedMpNum = 0;
    cout<<"Reject Outliers for Object3D "<<mTrackID<<"..."<<endl;
    int N_Mps = vpMps.size();
    vector<bool> status(N_Mps, true);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    vector<int> vMpindices;
    
    for(int i=0; i<N_Mps; i++)
    {
        MapPoint* pMp = vpMps[i];
        // if(pMp->isBad())
        // {
        //     cout<<"a MapPoint is bad, reject."<<endl;
        //     status[i] = false;
        //     continue;
        // }  
        cv::Mat PosW = pMp->GetWorldPos();
        pcl::PointXYZ p;
        p.x = PosW.at<float>(0);
        p.y = PosW.at<float>(1);
        p.z = PosW.at<float>(2);
        cloud->points.push_back(p);
        vMpindices.push_back(i);
    }

    vector<pcl::PointIndices> eceClustersIndices;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ece;
    tree->setInputCloud(cloud);
    ece.setInputCloud(cloud);
    ece.setClusterTolerance(0.1);
    ece.setMinClusterSize(1);
    ece.setMaxClusterSize(9999);
    ece.setSearchMethod(tree);
    ece.extract(eceClustersIndices);

    int cluster_num = eceClustersIndices.size();
    cout<<"Clustering "<<cluster_num<<" clusters, "<<endl;
    for(int j=0;j<cluster_num;j++)
    {
        int cluster_size = eceClustersIndices[j].indices.size();
        float ratio = (1.0*cluster_size)/(1.0*vMpindices.size());
        cout << "cluster "<<j<<" has "<<cluster_size<<" points, "<<"the ratio = "<<ratio<<endl;
        if(ratio < 0.1)
        {
            cout<<"reject cluster "<<j<<"."<<endl;
            for(auto index:eceClustersIndices[j].indices)
            {
                status[vMpindices[index]] = 0;
            }
        }
    }

    ReduceVector(vpMps, status);

}


//---------standard deviation of points----------//
void Object3D::RejectOutliersTEST5(vector<MapPoint*> &vpMps)
{
    mToLastRejectOutliers = 0;
    mnNewAddedMpNum = 0;
    //cout<<"Reject Outliers for Object3D "<<mTrackID<<"..."<<endl;
    int N_Mps = vpMps.size();
    vector<bool> status(N_Mps, true);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    float SquareSum = 0.0;
    vector<pair<int,cv::Mat>> vMpindicesPos;
    cv::Mat Center = cv::Mat::zeros(cv::Size(1,3),CV_32F);
    for(int i=0; i<N_Mps; i++)
    {
        MapPoint* pMp = vpMps[i];
        if(isMpBad(pMp))
        {
            status[i] = false;
            continue;
        }  
        cv::Mat PosW = pMp->GetWorldPos();
        Center = Center+PosW;
        float square = cv::norm(PosW);
        SquareSum = SquareSum+square*square;
        vMpindicesPos.push_back(make_pair(i, PosW));
    }
    int N_candidates = vMpindicesPos.size();
    Center = Center/(1.0*N_candidates);
    float CandidatesStdDev = sqrt(SquareSum/(1.0*N_candidates)-cv::norm(Center )*cv::norm(Center));
    for(int i=0;i<N_candidates;i++)
    {
        float distance = cv::norm(Center-vMpindicesPos[i].second);
        if(distance>3*CandidatesStdDev)
        {
            status[vMpindicesPos[i].first] = 0;
        }
    }

    ReduceVector(vpMps, status);
}

//------------label count--------------//
void Object3D::RejectOutliersTEST6(vector<MapPoint*> &vpMps)
{
    mToLastRejectOutliers = 0;
    mnNewAddedMpNum = 0;
    //cout<<"Reject Outliers for Object3D "<<mTrackID<<"..."<<endl;
    int N_Mps = vpMps.size();
    vector<bool> status(N_Mps, true);
    for(int i=0; i<N_Mps; i++)
    {
        MapPoint* pMp = vpMps[i];
        if(isMpBad(pMp))
        {
            status[i] = false;
            continue;
        }  

        if(pMp->GetLabelProb(mLabel)<0.1)
        {
            status[i] = false;
            continue;
        }

    }


    ReduceVector(vpMps, status);
}


//-------------ratio of clusters || standard deviation of points-------------//
void Object3D::RejectOutliersTEST7(vector<MapPoint*> &vpMps)
{
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    mToLastRejectOutliers = 0;
    mnNewAddedMpNum = 0;
    //cout<<"Reject Outliers for Object3D "<<mTrackID<<"..."<<endl;
    int N_Mps = vpMps.size();
    vector<bool> status(N_Mps, true);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    cv::Mat CandidatesCenter = cv::Mat::zeros(cv::Size(1,3),CV_32F);
    float SquareSum = 0.0;
    vector<pair<int,cv::Mat>> vMpindicesPos;
    for(int i=0; i<N_Mps; i++)
    {
        MapPoint* pMp = vpMps[i];
        if(isMpBad(pMp))
        {
            status[i] = false;
            continue;
        }  
        cv::Mat PosW = pMp->GetWorldPos();
        CandidatesCenter = CandidatesCenter+PosW;
        float square = cv::norm(PosW);
        SquareSum = SquareSum+square*square;
        pcl::PointXYZ p;
        p.x = PosW.at<float>(0);
        p.y = PosW.at<float>(1);
        p.z = PosW.at<float>(2);
        cloud->points.push_back(p);
        vMpindicesPos.push_back(make_pair(i,PosW));
    }
    int N_candidates = vMpindicesPos.size();
    CandidatesCenter = CandidatesCenter/(1.0*N_candidates);
    //simplified std dev calculation
    float CandidatesStdDev = sqrt(SquareSum/(1.0*N_candidates)-cv::norm(CandidatesCenter)*cv::norm(CandidatesCenter));

    //standard deviation
    // vector<float> CandidatesDistances;
    // for(int i=0;i<N_candidates;i++)
    // {
    //     float distance = cv::norm(CandidatesCenter-vMpindicesPos[i].second);
    //     CandidatesStdDev = CandidatesStdDev+distance*distance;
    //     CandidatesDistances.push_back(distance);
    // }
    // CandidatesStdDev = sqrt(CandidatesStdDev/(1.0*N_candidates));

    

    //DBSCAN
    vector<pcl::PointIndices> eceClustersIndices;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ece;
    tree->setInputCloud(cloud);
    ece.setInputCloud(cloud);
    ece.setClusterTolerance(0.1);//0.1 for indoor scenes, 1.0 for outdoor scenes
    ece.setMinClusterSize(1);
    ece.setMaxClusterSize(9999);
    ece.setSearchMethod(tree);
    ece.extract(eceClustersIndices);

    int cluster_num = eceClustersIndices.size();
    //cout<<"Clustering "<<cluster_num<<" clusters, "<<endl;

    // for(int i=0;i<N_candidates;i++)
    // {
    //     if(CandidatesDistances[i]>3*CandidatesStdDev)
    //     {
    //         status[vMpindicesPos[i].first] = false;;
    //     }
    // }
    

    for(int j=0;j<cluster_num;j++)
    {
        float ratio = (1.0*eceClustersIndices[j].indices.size())/(1.0*N_candidates);
        //cout << "cluster "<<j<<" has "<<eceClustersIndices[j].indices.size()<<" points, "<<"the ratio = "<<ratio<<endl;
        if(ratio < 0.1 && N_candidates > 15)
        {
            //cout<<"reject cluster "<<j<<"."<<endl;
            for(auto index: eceClustersIndices[j].indices)
            {
                status[vMpindicesPos[index].first] = false;
                //status[index] = 0;
            }
        }
        else
        {
            for(auto index:eceClustersIndices[j].indices)
            {
                float dist = cv::norm(vMpindicesPos[index].second-CandidatesCenter);
                if(dist>3*CandidatesStdDev)
                {
                    status[vMpindicesPos[index].first] = false;
                }
            }
        }
    }

    ReduceVector(vpMps, status);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
    //cout<<"reject outliers use "<<ttrack<<" s"<<endl;
}

void Object3D::CalculateCenterW()
{
    
    if(!mvpMapPoints.empty())
    {
        Eigen::Vector3f CenterW = Eigen::Vector3f::Zero();
        for(auto p:mvpMapPoints)
        {
            cv::Mat PosW = p->GetWorldPos();
            Eigen::Vector3f P(PosW.at<float>(0), PosW.at<float>(1), PosW.at<float>(2));
            CenterW = CenterW+P;
        }
        mCenterW = CenterW/(1.0*mvpMapPoints.size());
    }
}

void Object3D::CalculateSize()
{
    
    if(!mvpMapPoints.empty())
    {
        vector<float> Xs, Ys, Zs;
        for(auto p:mvpMapPoints)
        {
            cv::Mat PosW = p->GetWorldPos();
            Xs.push_back(PosW.at<float>(0));
            Ys.push_back(PosW.at<float>(1));
            Zs.push_back(PosW.at<float>(2));
        }
        sort(Xs.begin(),Xs.end());
        sort(Ys.begin(),Ys.end());
        sort(Zs.begin(),Zs.end());

        mMaxXw = Xs.back(); mMinXw = Xs.front();
        mMaxYw = Ys.back(); mMinYw = Ys.front();
        mMaxZw = Zs.back(); mMinZw = Zs.front();
    }
}

void Object3D::CalculateCenterAndSize()
{
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();    
    if(!mvpMapPoints.empty())
    {
        Eigen::Vector3f CenterW = Eigen::Vector3f::Zero();
        vector<float> Xs, Ys, Zs;
        for(auto p:mvpMapPoints)
        {
            cv::Mat PosW = p->GetWorldPos();
            Eigen::Vector3f P(PosW.at<float>(0), PosW.at<float>(1), PosW.at<float>(2));
            CenterW = CenterW+P;
            Xs.push_back(PosW.at<float>(0));
            Ys.push_back(PosW.at<float>(1));
            Zs.push_back(PosW.at<float>(2));
        }
        mCenterW = CenterW/(1.0*mvpMapPoints.size());
        sort(Xs.begin(),Xs.end());
        sort(Ys.begin(),Ys.end());
        sort(Zs.begin(),Zs.end());

        mMaxXw = Xs.back(); mMinXw = Xs.front();
        mMaxYw = Ys.back(); mMinYw = Ys.front();
        mMaxZw = Zs.back(); mMinZw = Zs.front();
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
        //cout<<"calculate size and center use "<<ttrack<<" s"<<endl;
    }
}






}
