#include "ObjectMatcher.h"
#include "ObjectTypes.h"
#include "opencv2/opencv.hpp"
#include "MapPoint.h"
#include "Converter.h"
#include <float.h>



namespace ORB_SLAM2{
 

const int ObjectMatcher::TH_HIGH = 100;
const int ObjectMatcher::TH_LOW = 50;
const int ObjectMatcher::HISTO_LENGTH = 30;

ObjectMatcher::ObjectMatcher()
{}

//***********for visualization**********//
cv::Mat TwoImgFusing(cv::Mat &img1, cv::Mat &img2);
cv::Mat TwoImgFusing(cv::Mat &img1, cv::Mat &img2)
{
    int ROW = img1.rows;
    int COL = img1.cols;
    cv::Mat res = cv::Mat(ROW*2,COL,CV_8UC3);

    for(int row=0;row<ROW;row++)
    {
        for(int col=0;col<COL;col++)
        {
            res.ptr<uchar>(row)[3*col] = img1.ptr<uchar>(row)[3*col];
            res.ptr<uchar>(row)[3*col+1] = img1.ptr<uchar>(row)[3*col+1];
            res.ptr<uchar>(row)[3*col+2] = img1.ptr<uchar>(row)[3*col+2];
            res.ptr<uchar>(ROW+row)[3*col] = img2.ptr<uchar>(row)[3*col];
            res.ptr<uchar>(ROW+row)[3*col+1] = img2.ptr<uchar>(row)[3*col+1];
            res.ptr<uchar>(ROW+row)[3*col+2] = img2.ptr<uchar>(row)[3*col+2];
        }
    }
    return res;
}
//***********for visualization**********//




void ObjectMatcher::MatchTwoFrame(Frame &PreF, Frame &CurF)
{

    vector<Object2D> vPreObj2Ds = PreF.mvObject2Ds;
    int N_PreObj2D = vPreObj2Ds.size();
    vector<Object3D*> vPreObj3DS = PreF.mvpObject3Ds;
    
    vector<Object2D> vCurObj2Ds = CurF.mvObject2Ds;
    int N_CurObj2D = vCurObj2Ds.size();
    vector<MapPoint*> vpCurMps = CurF.mvpMapPoints;
    
    
    for(int idx_pre; idx_pre<N_PreObj2D; idx_pre++)
    {
        Object3D* pObj3D = vPreObj3DS[idx_pre];
        if(pObj3D)
        {
            Object2D PreObj2D = vPreObj2Ds[idx_pre];
            
            //For Appearance Similarity Check
            map<int, float> CurObj2DIdx_Similarity;
            cv::Mat PreAppearanceVec = PreObj2D.mHSVHistogram;

            //For 3D-2D MapPoint-KeyPoint Matches Check
            vector<MapPoint*> vpPreObj3DMps = pObj3D->mvpMapPoints;
            vector<MapPoint*>::iterator it_start = vpPreObj3DMps.begin();
            vector<MapPoint*>::iterator it_end = vpPreObj3DMps.end();
            map<int,vector<pair<int,int>>> CurObj2DIdx_Match3D2D;//Match.first=Obj3DMpIdx, Match.second=Obj2DKpIdx
            map<int, float> CurObj2DIdx_Match3D2D_RatiowrtKps;
            map<int, float> CurObj2DIdx_Match3D2D_RatiowrtMps;
            map<int, float> CurObj2DIdx_Match3D2D_RatioIoU;

            //For 2D Bounding Box IoU Check
            map<int, float> CurObj2DIdx_BBox2DIoU;
            float PreBBoxArea = PreObj2D.w*PreObj2D.h;
            vector<float> PreBBox = {PreObj2D.x, PreObj2D.y, PreObj2D.x+PreObj2D.w, PreObj2D.y+PreObj2D.h}; //[MinX, MinY, MaxX, MaxY]

            //For 2D-2D ORB Matches Check
            const DBoW2::FeatureVector &vFeatVecPreObj2D  = PreObj2D.mFeatVec;
            map<int, vector<pair<int,int>>>  CurObj2DIdx_Match2D2D; //Match.first=preIdx, Match.second=curIdx
            bool bCheckOrientation = false;
            float fNNratio = 0.6;
            map<int, float> CurObj2DIdx_Match2D2D_RatiowrtPreKps;
            map<int, float> CurObj2DIdx_Match2D2D_RatiowrtCurKps;
            map<int, float> CurObj2DIdx_Match2D2D_RatioIoU;

            

            
            for(int idx_cur=0; idx_cur<N_CurObj2D; idx_cur++)
            {
                if(CurF.mvpObject3Ds[idx_cur])
                    continue;               
                Object2D CurObj2D = vCurObj2Ds[idx_cur];
                if(CurObj2D.label!=pObj3D->mLabel)
                    continue;


                ///////////////Appearance Similarity Check///////////
                cv::Mat CurAppearanceVec = CurObj2D.mHSVHistogram;
                float CosineSimilarity = CalCosineSimilarity(PreAppearanceVec, CurAppearanceVec);
                CurObj2DIdx_Similarity[idx_cur] = CosineSimilarity;
                ////////////////Appearance Similarity Check END/////////////
               
                ///////////////3D-2D MapPoint-KeyPoint Matches Check/////////////
                // vector<int> vFrameKpIndices = CurObj2D.mvFrameKpIndices;
                // vector<pair<int,int>> TempMatch3D2D;
                // for(int i=0;i<vFrameKpIndices.size();i++)
                // {
                //     MapPoint* pMp = vpCurMps[vFrameKpIndices[i]];
                //     vector<MapPoint*>::iterator it_find = find(it_start, it_end, pMp);
                //     if(it_find!=it_end)
                //     {
                //         int Obj3DMpIdx = it_find-it_start;
                //         TempMatch3D2D.push_back(make_pair(Obj3DMpIdx, i));       
                //     }
                // }
                // CurObj2DIdx_Match3D2D[idx_cur] =  TempMatch3D2D;
                // CurObj2DIdx_Match3D2D_RatiowrtKps[idx_cur] = (1.0*TempMatch3D2D.size())/vFrameKpIndices.size();
                // CurObj2DIdx_Match3D2D_RatiowrtMps[idx_cur] = (1.0*TempMatch3D2D.size())/vpPreObj3DMps.size();
                // CurObj2DIdx_Match3D2D_RatioIoU[idx_cur] = (1.0*TempMatch3D2D.size())/(vpPreObj3DMps.size()+vFrameKpIndices.size());
                ///////////////3D-2D MapPoint-KeyPoint Matches Check END/////////////

                ///////////////2D BBox Check/////////////
                vector<float> CurBBox = {CurObj2D.x, CurObj2D.y, CurObj2D.x+CurObj2D.w, CurObj2D.y+CurObj2D.h}; 
                float CurBBoxArea = CurObj2D.w*CurObj2D.h;
                float x1 = max(CurBBox[0], PreBBox[0]);//intersection minX
                float x2 = min(CurBBox[2], PreBBox[2]);//intersection maxX
                float y1 = max(CurBBox[1], PreBBox[1]);//intersection minY
                float y2 = min(CurBBox[3], PreBBox[3]);//intersection maxY
                float BBox2DIoU = 0.0;
                if(x1>=x2 || y1>=y2)
                {
                    BBox2DIoU = 0.0;
                }
                else
                {
                    BBox2DIoU = (x2-x1)*(y2-y1);
                    BBox2DIoU = BBox2DIoU/(PreBBoxArea+CurBBoxArea-BBox2DIoU);
                    CurObj2DIdx_BBox2DIoU[idx_cur] = BBox2DIoU;
                }
                ///////2D BBox Check END////////////

                ////////////////2D-2D ORB Matches Check/////////////
                // const DBoW2::FeatureVector &vFeatVecCurObj2D = CurObj2D.mFeatVec;
                // int nORBMatches=0;
                // vector<pair<int,int>> TempMatch2D2D;
                // vector<cv::Point2f> vPreMatched;
                // vector<cv::Point2f> vCurMatched;
                // vector<int> rotHist[HISTO_LENGTH];
                // for(int i=0;i<HISTO_LENGTH;i++)
                //     rotHist[i].reserve(500);
                // const float factor = 1.0f/HISTO_LENGTH;
                // DBoW2::FeatureVector::const_iterator Preit = vFeatVecPreObj2D.begin();
                // DBoW2::FeatureVector::const_iterator Curit = vFeatVecCurObj2D.begin();
                // DBoW2::FeatureVector::const_iterator Preend = vFeatVecPreObj2D.end();
                // DBoW2::FeatureVector::const_iterator Curend = vFeatVecCurObj2D.end();
                // while(Preit != Preend && Curit != Curend)
                // {
                //     if(Preit->first == Curit->first)
                //     {
                //         const vector<unsigned int> vIndicesPreObj2D = Preit->second;
                //         const vector<unsigned int> vIndicesCurObj2D = Curit->second;
                //         for(size_t iPre=0; iPre<vIndicesPreObj2D.size(); iPre++)
                //         {
                //             const unsigned int realIdxPre = vIndicesPreObj2D[iPre];
                //             // ObjectMapPoint* pObjMP = vpMapPointsPre[realIdxPre];
                //             // if(!pObjMP)
                //             //     continue;
                //             const cv::Mat &dPre= PreObj2D.mvDescriptors[realIdxPre];
                //             int bestDist1=256;
                //             int bestDist2=256;
                //             int bestIdxCur =-1 ;
                //             for(size_t iCur=0; iCur<vIndicesCurObj2D.size(); iCur++)
                //             {
                //                 const unsigned int realIdxCur = vIndicesCurObj2D[iCur];
                //                 // if(vpMapPointMatches[realIdxF])
                //                 //     continue;
                //                 const cv::Mat &dCur = CurObj2D.mvDescriptors[realIdxCur];
                //                 const int dist =  DescriptorDistance(dPre,dCur);
                //                 if(dist<bestDist1)
                //                 {
                //                     bestDist2=bestDist1;
                //                     bestDist1=dist;
                //                     bestIdxCur=realIdxCur;
                //                 }
                //                 else if(dist<bestDist2)
                //                 {
                //                     bestDist2=dist;
                //                 }
                //             }
                //             if(bestDist1<=TH_LOW)
                //             {
                //                 if(static_cast<float>(bestDist1)<fNNratio*static_cast<float>(bestDist2))
                //                 {
                //                     //vpObjectMapPointMatches[bestIdxCur]=pObjMP;
                //                     TempMatch2D2D.push_back(make_pair(realIdxPre, bestIdxCur));
                //                     vPreMatched.push_back(PreObj2D.mvKeyPoints[realIdxPre].pt);
                //                     vCurMatched.push_back(CurObj2D.mvKeyPoints[bestIdxCur].pt);
                //                     const cv::KeyPoint &kp = PreObj2D.mvKeyPoints[realIdxPre];
                //                     if(bCheckOrientation)
                //                     {
                //                         cv::KeyPoint &Curkp = CurObj2D.mvKeyPoints[bestIdxCur];
                //                         float rot = kp.angle-Curkp.angle;
                //                         if(rot<0.0)
                //                             rot+=360.0f;
                //                         int bin = round(rot*factor);
                //                         if(bin==HISTO_LENGTH)
                //                             bin=0;
                //                         assert(bin>=0 && bin<HISTO_LENGTH);
                //                         rotHist[bin].push_back(bestIdxCur);
                //                     }
                //                     nORBMatches++;
                //                 }
                //             }
                //         }
                //         Preit++;
                //         Curit++;
                //     }
                //     else if(Preit->first < Curit->first)
                //     {
                //         Preit = vFeatVecPreObj2D.lower_bound(Curit->first);
                //     }
                //     else
                //     {
                //         Curit = vFeatVecCurObj2D.lower_bound(Preit->first);
                //     }
                // }
                // if(bCheckOrientation)
                // {
                //     int ind1=-1;
                //     int ind2=-1;
                //     int ind3=-1;
                //     ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);
                //     for(int i=0; i<HISTO_LENGTH; i++)
                //     {
                //         if(i==ind1 || i==ind2 || i==ind3)
                //             continue;
                //         for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                //         {
                //             // vpMapPointMatches[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                //             nORBMatches--;
                //         }
                //     }
                // }
                // if(vPreMatched.size()>0)
                // {
                //     vector<uchar> status = vector<uchar>(vPreMatched.size(), 1);
                //     RejectOutliersByRANSAC(vPreMatched, vCurMatched, status);
                //     ReduceVector(TempMatch2D2D, status);
                // }
                // CurObj2DIdx_Match2D2D[idx_cur] = TempMatch2D2D;
                // CurObj2DIdx_Match2D2D_RatiowrtPreKps[idx_cur] = (1.0*TempMatch2D2D.size())/(1.0*CurObj2D.mvKeyPoints.size());
                // CurObj2DIdx_Match2D2D_RatiowrtCurKps[idx_cur] = (1.0*TempMatch2D2D.size())/(1.0*PreObj2D.mvKeyPoints.size());
                // CurObj2DIdx_Match2D2D_RatioIoU[idx_cur] = (1.0*TempMatch2D2D.size())/(1.0*(PreObj2D.mvKeyPoints.size()+CurObj2D.mvKeyPoints.size()));
                //////////////ORB Matches Check END/////////////
            }

            ///////////////---------------------Analyze-------------------////////////////

            //-------------Appearance Similarity-------------//
            //cout<<"*********Similarity Check Analyze*******"<<endl;
            float MaxSimilarity = 0.0;
            int MaxSimilarityCurObj2DIdx = -1;
            if(!CurObj2DIdx_Similarity.empty())
            {
                vector<pair<int,float>> vCurObj2DIdx_Similarity = vector<pair<int,float>>(CurObj2DIdx_Similarity.begin(), CurObj2DIdx_Similarity.end());
                sort(vCurObj2DIdx_Similarity.begin(), vCurObj2DIdx_Similarity.end(), [&] (pair<int,float> a, pair<int,float> b) {return a.second<b.second;});

                MaxSimilarity = vCurObj2DIdx_Similarity.back().second;
                MaxSimilarityCurObj2DIdx = vCurObj2DIdx_Similarity.back().first;
                // for(auto temp:vCurObj2DIdx_Similarity)
                // {
                //     cout<<"Appearance Similarity: Object3D "<<pObj3D->mTrackID<<" <-> Object2D "<<temp.first<<" ="<<temp.second<<endl;
                // }
                // cout<<"Max Appearance Similarity: Object3D "<<pObj3D->mTrackID<<" <-> Object2D "<<MaxSimilarityCurObj2DIdx<<" ="<<MaxSimilarity<<endl;
            }

            //--------END----------//

            //------------------2D BBox-------------//
            //cout<<"*********2D BBox IoU Check Analyze*******"<<endl;
            float MaxBBox2DIoU = 0.0;
            int MaxBBox2DIoUCurObj2DIdx = -1;
            if(!CurObj2DIdx_BBox2DIoU.empty())
            {
                vector<pair<int,float>> vCurObj2DIdx_BBox2DIoU = vector<pair<int,float>>(CurObj2DIdx_BBox2DIoU.begin(), CurObj2DIdx_BBox2DIoU.end());
                sort(vCurObj2DIdx_BBox2DIoU.begin(), vCurObj2DIdx_BBox2DIoU.end(), [&] (pair<int,float> a, pair<int,float> b) {return a.second<b.second;});

                MaxBBox2DIoU = vCurObj2DIdx_BBox2DIoU.back().second;
                MaxBBox2DIoUCurObj2DIdx = vCurObj2DIdx_BBox2DIoU.back().first;
                // for(auto temp:vCurObj2DIdx_BBox2DIoU)
                // {
                //     cout<<"2D BBox IoU: Object3D "<<pObj3D->mTrackID<<" <-> Object2D "<<temp.first<<" ="<<temp.second<<endl;
                // }
                // cout<<"Max 2D BBox IoU: Object3D "<<pObj3D->mTrackID<<" <-> Object2D "<<MaxBBox2DIoUCurObj2DIdx<<" ="<<MaxBBox2DIoU<<endl;
            }



            //----------END-------//


            //-------------MapPoint-KeyPoint 3D-2D Match-----------//
            //cout<<"*********3D-2D Match Analyze*******"<<endl;
            // int MaxMatch3D2DNum = 0;
            // int MaxMatch3D2DCurObj2DIdx = -1;
            // if(!CurObj2DIdx_Match3D2D.empty())
            // {
            //     vector<pair<int,vector<pair<int,int>>>> vCurObj2DIdx_Match3D2D = vector<pair<int, vector<pair<int,int>>>>(CurObj2DIdx_Match3D2D.begin(), CurObj2DIdx_Match3D2D.end());
            //     sort(vCurObj2DIdx_Match3D2D.begin(), vCurObj2DIdx_Match3D2D.end(), [&] (pair<int, vector<pair<int,int>>> a, pair<int, vector<pair<int,int>>> b) {return a.second.size()<b.second.size();});        
            //     int MaxMatch3D2DNum = vCurObj2DIdx_Match3D2D.back().second.size();
            //     int MaxMatch3D2DCurObj2DIdx = vCurObj2DIdx_Match3D2D.back().first;
            //     for(auto temp:vCurObj2DIdx_Match3D2D)
            //     {
            //         cout<<"3D-2D Match: Object3D "<<pObj3D->mTrackID<<" <-> Object2D "<<temp.first<<" ="<<temp.second.size()<<endl;
            //     }
            //     cout<<"Max 3D-2D Match: Object3D "<<pObj3D->mTrackID<<" <-> Object2D "<<MaxMatch3D2DCurObj2DIdx<<" ="<<MaxMatch3D2DNum<<endl;       
            // }
            // float MaxMatch3D2DRatiowrtKps = 0.0;
            // int MaxMatch3D2DRatiowrtKpsCurObj2DIdx = -1;
            // if(!CurObj2DIdx_Match3D2D_RatiowrtKps.empty())
            // {
            //     vector<pair<int,float>> vCurObj2DIdx_Match3D2D_RatiowrtKps = vector<pair<int, float>>(CurObj2DIdx_Match3D2D_RatiowrtKps.begin(), CurObj2DIdx_Match3D2D_RatiowrtKps.end());
            //     sort(vCurObj2DIdx_Match3D2D_RatiowrtKps.begin(), vCurObj2DIdx_Match3D2D_RatiowrtKps.end(), [&] (pair<int, float> a, pair<int,float> b) {return a.second<b.second;});
            //     MaxMatch3D2DRatiowrtKps = vCurObj2DIdx_Match3D2D_RatiowrtKps.back().second;
            //     MaxMatch3D2DRatiowrtKpsCurObj2DIdx = vCurObj2DIdx_Match3D2D_RatiowrtKps.back().first;
            //     for(auto temp:vCurObj2DIdx_Match3D2D_RatiowrtKps)
            //     {
            //         cout<<"3D-2D Match ratio wrt Kps: Object3D "<<pObj3D->mTrackID<<" <-> Object2D "<<temp.first<<" ="<<temp.second<<endl;
            //     }
            //     cout<<"Max 3D-2D Match ratio wrt Kps: Object3D "<<pObj3D->mTrackID<<" <-> Object2D "<<MaxMatch3D2DRatiowrtKpsCurObj2DIdx<<" ="<<MaxMatch3D2DRatiowrtKps<<endl;
            // }
            // float MaxMatch3D2DRatiowrtMps = 0.0;
            // int MaxMatch3D2DRatiowrtMpsCurObj2DIdx = -1;
            // if(!CurObj2DIdx_Match3D2D_RatiowrtMps.empty())
            // {
            //     vector<pair<int,float>> vCurObj2DIdx_Match3D2D_RatiowrtMps = vector<pair<int, float>>(CurObj2DIdx_Match3D2D_RatiowrtMps.begin(), CurObj2DIdx_Match3D2D_RatiowrtMps.end());
            //     sort(vCurObj2DIdx_Match3D2D_RatiowrtMps.begin(), vCurObj2DIdx_Match3D2D_RatiowrtMps.end(), [&] (pair<int, float> a, pair<int,float> b) {return a.second<b.second;});
            //     MaxMatch3D2DRatiowrtMps = vCurObj2DIdx_Match3D2D_RatiowrtMps.back().second;
            //     MaxMatch3D2DRatiowrtMpsCurObj2DIdx = vCurObj2DIdx_Match3D2D_RatiowrtMps.back().first;
            //     for(auto temp:vCurObj2DIdx_Match3D2D_RatiowrtMps)
            //     {
            //         cout<<"3D-2D Match ratio wrt Mps: Object3D "<<pObj3D->mTrackID<<" <-> Object2D "<<temp.first<<" ="<<temp.second<<endl;
            //     }
            //     cout<<"Max 3D-2D Match ratio wrt Mps: Object3D "<<pObj3D->mTrackID<<" <-> Object2D "<<MaxMatch3D2DRatiowrtMpsCurObj2DIdx<<" ="<<MaxMatch3D2DRatiowrtMps<<endl;
            // }
            // float MaxMatch3D2DRatioIoU= 0.0;
            // int MaxMatch3D2DRatioIoUCurObj2DIdx = -1;
            // if(!CurObj2DIdx_Match3D2D_RatioIoU.empty())
            // {
            //     vector<pair<int,float>> vCurObj2DIdx_Match3D2D_RatioIoU = vector<pair<int, float>>(CurObj2DIdx_Match3D2D_RatioIoU.begin(), CurObj2DIdx_Match3D2D_RatioIoU.end());
            //     sort( vCurObj2DIdx_Match3D2D_RatioIoU.begin(),  vCurObj2DIdx_Match3D2D_RatioIoU.end(), [&] (pair<int, float> a, pair<int,float> b) {return a.second<b.second;});
            //     MaxMatch3D2DRatioIoU= vCurObj2DIdx_Match3D2D_RatioIoU.back().second;
            //     MaxMatch3D2DRatioIoUCurObj2DIdx = vCurObj2DIdx_Match3D2D_RatioIoU.back().first;
            //     for(auto temp:vCurObj2DIdx_Match3D2D_RatioIoU)
            //     {
            //         cout<<"3D-2D Match ratio IoU: Object3D "<<pObj3D->mTrackID<<" <-> Object2D "<<temp.first<<" ="<<temp.second<<endl;
            //     }
            //     cout<<"Max 3D-2D Match ratio IoU: Object3D "<<pObj3D->mTrackID<<" <-> Object2D "<<MaxMatch3D2DRatioIoUCurObj2DIdx<<" ="<<MaxMatch3D2DRatioIoU<<endl;
            // }
            //--------END------//
            //-----------ORB 2D-2D Match--------------//
            // cout<<"*********2D-2D Match Check Analyze*******"<<endl;
            // int MaxMatch2D2DNum = 0;
            // int MaxMatch2D2DCurObj2DIdx = -1;
            // if(!CurObj2DIdx_Match2D2D.empty())
            // {
            //     vector<pair<int, vector<pair<int,int>>>> vCurObj2DIdx_Match2D2D = vector<pair<int, vector<pair<int,int>>>>(CurObj2DIdx_Match2D2D.begin(), CurObj2DIdx_Match2D2D.end());
            //     sort(vCurObj2DIdx_Match2D2D.begin(), vCurObj2DIdx_Match2D2D.end(), [&] (pair<int, vector<pair<int,int>>> a, pair<int, vector<pair<int,int>>> b) {return a.second.size()<b.second.size();});
            //     MaxMatch2D2DNum = vCurObj2DIdx_Match2D2D.back().second.size();
            //     MaxMatch2D2DCurObj2DIdx = vCurObj2DIdx_Match2D2D.back().first;
            //     for(auto temp:vCurObj2DIdx_Match2D2D)
            //     {
            //         cout<<"2D-2D Match Number: Object3D "<<pObj3D->mTrackID<<" <-> Object2D "<<temp.first<<" ="<<temp.second.size()<<endl;
            //     }
            //     cout<<"Max 2D-2D Match Number: Object3D "<<pObj3D->mTrackID<<" <-> Object2D "<<MaxMatch2D2DCurObj2DIdx<<" ="<<MaxMatch2D2DNum<<endl;
            // }
            // float MaxMatch2D2DRatiowrtPreKps = 0.0;
            // int MaxMatch2D2DRatiowrtPreKpsCurObj2DIdx = -1;
            // if(!CurObj2DIdx_Match2D2D.empty())
            // {
            //     vector<pair<int,float>> vCurObj2DIdx_Match2D2D_RatiowrtPreKps = vector<pair<int, float>>(CurObj2DIdx_Match2D2D_RatiowrtPreKps.begin(), CurObj2DIdx_Match2D2D_RatiowrtPreKps.end());
            //     sort(vCurObj2DIdx_Match2D2D_RatiowrtPreKps.begin(), vCurObj2DIdx_Match2D2D_RatiowrtPreKps.end(), [&] (pair<int, float> a, pair<int,float> b) {return a.second<b.second;});
            //     MaxMatch2D2DRatiowrtPreKps = vCurObj2DIdx_Match2D2D_RatiowrtPreKps.back().second;
            //     MaxMatch2D2DRatiowrtPreKpsCurObj2DIdx = vCurObj2DIdx_Match2D2D_RatiowrtPreKps.back().first;
            //     for(auto temp:vCurObj2DIdx_Match2D2D_RatiowrtPreKps)
            //     {
            //         cout<<"2D-2D Match ratio wrt PreKps: Object3D "<<pObj3D->mTrackID<<" <-> Object2D "<<temp.first<<" ="<<temp.second<<endl;
            //     }
            //     cout<<"Max 2D-2D Match ratio wrt PreKps: Object3D "<<pObj3D->mTrackID<<" <-> Object2D "<<MaxMatch2D2DRatiowrtPreKpsCurObj2DIdx<<" ="<<MaxMatch2D2DRatiowrtPreKps<<endl;
            // }
            // float MaxMatch2D2DRatiowrtCurKps = 0.0;
            // int MaxMatch2D2DRatiowrtCurKpsCurObj2DIdx = -1;
            // if(!CurObj2DIdx_Match2D2D_RatiowrtCurKps.empty())
            // {
            //     vector<pair<int,float>> vCurObj2DIdx_Match2D2D_RatiowrtCurKps = vector<pair<int, float>>(CurObj2DIdx_Match2D2D_RatiowrtCurKps.begin(), CurObj2DIdx_Match2D2D_RatiowrtCurKps.end());
            //     sort(vCurObj2DIdx_Match2D2D_RatiowrtCurKps.begin(), vCurObj2DIdx_Match2D2D_RatiowrtCurKps.end(), [&] (pair<int, float> a, pair<int,float> b) {return a.second<b.second;});
            //     MaxMatch2D2DRatiowrtCurKps = vCurObj2DIdx_Match2D2D_RatiowrtCurKps.back().second;
            //     MaxMatch2D2DRatiowrtCurKpsCurObj2DIdx = vCurObj2DIdx_Match2D2D_RatiowrtCurKps.back().first;
            //     for(auto temp:vCurObj2DIdx_Match2D2D_RatiowrtCurKps)
            //     {
            //         cout<<"2D-2D Match ratio wrt CurMps: Object3D "<<pObj3D->mTrackID<<" <-> Object2D "<<temp.first<<" ="<<temp.second<<endl;
            //     }
            //     cout<<"Max 2D-2D Match ratio wrt CurMps: Object3D "<<pObj3D->mTrackID<<" <-> Object2D "<<MaxMatch2D2DRatiowrtCurKpsCurObj2DIdx<<" ="<<MaxMatch2D2DRatiowrtCurKps<<endl;
            // }
            // float MaxMatch2D2DRatioIoU = 0.0;
            // int MaxMatch2D2DRatioIoUCurObj2DIdx = -1;
            // if(!CurObj2DIdx_Match2D2D_RatioIoU.empty())
            // {
            //     vector<pair<int,float>> vCurObj2DIdx_Match2D2D_RatioIoU = vector<pair<int, float>>(CurObj2DIdx_Match2D2D_RatioIoU.begin(), CurObj2DIdx_Match2D2D_RatioIoU.end());
            //     sort(vCurObj2DIdx_Match2D2D_RatioIoU.begin(), vCurObj2DIdx_Match2D2D_RatioIoU.end(), [&] (pair<int, float> a, pair<int,float> b) {return a.second<b.second;});
            //     MaxMatch2D2DRatioIoU = vCurObj2DIdx_Match2D2D_RatioIoU.back().second;
            //     MaxMatch2D2DRatioIoUCurObj2DIdx = vCurObj2DIdx_Match2D2D_RatioIoU.back().first;
            //     for(auto temp:vCurObj2DIdx_Match2D2D_RatioIoU)
            //     {
            //         cout<<"2D-2D Match ratio IoU: Object3D "<<pObj3D->mTrackID<<" <-> Object2D "<<temp.first<<" ="<<temp.second<<endl;
            //     }
            //     cout<<"Max 2D-2D Match ratio IoU: Object3D "<<pObj3D->mTrackID<<" <-> Object2D "<<MaxMatch2D2DRatioIoUCurObj2DIdx<<" ="<<MaxMatch2D2DRatioIoU<<endl;
            // }
            //--------END---------//

            //---------------TEST---------------//
            if(MaxSimilarity > 0.8 && MaxSimilarityCurObj2DIdx>=0 && CurObj2DIdx_BBox2DIoU[MaxSimilarityCurObj2DIdx]>0.5)
            {
                //cout<<"Assign Object3D "<<pObj3D->mTrackID<<" to Object2D "<<MaxSimilarityCurObj2DIdx<<endl;
                CurF.mvpObject3Ds[MaxSimilarityCurObj2DIdx] = pObj3D;
                CurF.mvObject2Ds[MaxSimilarityCurObj2DIdx].track_id = pObj3D->mTrackID;
            }

                             
        }
    }
}

void ObjectMatcher::MatchMapToFrame(vector<Object3D*> vpObject3Ds, Frame &F)
{

    cv::Mat Tcw = F.mTcw;
    cv::Mat Ow = F.GetOw();

    vector<Object2D> vObj2Ds = F.mvObject2Ds;
    int N_Obj2D = vObj2Ds.size();
    vector<MapPoint*> vpFrameMps = F.mvpMapPoints;
    set<Object3D*> spMatchedFrameObj3Ds;
    for(int i=0;i<N_Obj2D;i++)
    {
        Object3D *pObj3D = F.mvpObject3Ds[i];
        if(pObj3D)
        {
            spMatchedFrameObj3Ds.insert(pObj3D);
        }
    }

    
    vector<Object3D*> vpObj3Ds = vpObject3Ds;
    int N_Obj3D = vpObj3Ds.size();
    
    for(int idx_3d = 0; idx_3d<N_Obj3D; idx_3d++)
    {
        Object3D *pObj3D = vpObj3Ds[idx_3d];
        if(pObj3D)
        {
            if(spMatchedFrameObj3Ds.find(pObj3D)!=spMatchedFrameObj3Ds.end())
                continue;

            //For Common Checks 
            vector<MapPoint*> vpObj3DMps = pObj3D->mvpMapPoints;
            vector<ObjectObservation> vObs = pObj3D->mvObservations;   

            //For Appearance Similarity Check
            cv::Mat CurCenterW = Converter::toCvMat(pObj3D->mCenterW);
            map<int, float> FrameObj2DIdx_Similarity;
       
            cv::Mat CurObsView = CurCenterW-Ow;
            CurObsView = norm(CurObsView);

            //For 3D-2D MapPoint-KeyPoint Matches Check
            vector<MapPoint*>::iterator it_start = vpObj3DMps.begin();
            vector<MapPoint*>::iterator it_end = vpObj3DMps.end();
            map<int,vector<pair<int,int>>> FrameObj2DIdx_Match3D2D;//Match.first=Obj3DMpIdx, Match.second=Obj2DKpIdx
            map<int, float> FrameObj2DIdx_Match3D2D_RatiowrtKps;
            map<int, float> FrameObj2DIdx_Match3D2D_RatiowrtMps;
            map<int, float> FrameObj2DIdx_Match3D2D_RatioIoU;

            //For Position Distance Check
            map<int, float> FrameObj2DIdx_MeanDist;
            map<int, float> FrameObj2DIdx_MinDist;

            //For BBox3D Check
            vector<float> ObjBBox3D = {pObj3D->mMaxXw, pObj3D->mMaxYw, pObj3D->mMaxZw, pObj3D->mMinXw, pObj3D->mMinYw, pObj3D->mMinZw};
            map<int,float> FrameObj2DIdx_BBox3DIoU;
            map<int,float> FrameObj2DIdx_BBox3DOverlapRatio;





            for(int idx_2d=0; idx_2d<N_Obj2D;idx_2d++)
            {
                if(F.mvpObject3Ds[idx_2d])
                    continue;

                Object2D Obj2D = vObj2Ds[idx_2d];

                if(Obj2D.label != pObj3D->mLabel)
                    continue;


            
                ///////////////Appearance Similarity Check///////////
                cv::Mat Obj2DAppearanceVec = Obj2D.mHSVHistogram;
                float CosineSimilarity = 0.0;
                for(auto obs: vObs)
                {
                    cv::Mat OldCenterW = Converter::toCvMat(obs.ObjectCenterW);
                    cv::Mat OldObsCamPoseTcw = obs.ObsCamPoseTcw.clone();
                    cv::Mat OldRcw = OldObsCamPoseTcw.colRange(0,3).rowRange(0,3);
                    cv::Mat Oldtcw = OldObsCamPoseTcw.colRange(2,3).rowRange(0,3);
                    cv::Mat OldOw = -OldRcw.t()*Oldtcw;
                    cv::Mat OldObsView = OldCenterW-OldOw;
                    OldObsView = OldObsView/norm(OldObsView);
                    float AngleCos = CurObsView.dot(OldObsView);
                    //if(AngleCos > 0)
                    {
                        cv::Mat ObservedAppearanceVec = obs.AppearanceVec;
                        float temp_similarity = CalCosineSimilarity(ObservedAppearanceVec, Obj2DAppearanceVec);
                        if(temp_similarity > CosineSimilarity)
                        {
                            CosineSimilarity = temp_similarity;
                        }
                    }

                }

                FrameObj2DIdx_Similarity[idx_2d] = CosineSimilarity;
                ////////////////Appearance Similarity Check END/////////////

                vector<int> vFrameKpIndices = Obj2D.mvFrameKpIndices;
                ///////////////3D-2D MapPoint-KeyPoint Matches Check/////////////
                vector<pair<int,int>> TempMatch3D2D;
                for(int i=0;i<vFrameKpIndices.size();i++)
                {
                    MapPoint* pMp = vpFrameMps[vFrameKpIndices[i]];
                    vector<MapPoint*>::iterator it_find = find(it_start, it_end, pMp);
                    if(it_find!=it_end)
                    {
                        
                        int Obj3DMpIdx = it_find-it_start;
                        TempMatch3D2D.push_back(make_pair(Obj3DMpIdx, i));
                    }
                }
                FrameObj2DIdx_Match3D2D[idx_2d] =  TempMatch3D2D;
                FrameObj2DIdx_Match3D2D_RatiowrtKps[idx_2d] = (1.0*TempMatch3D2D.size())/vFrameKpIndices.size();
                FrameObj2DIdx_Match3D2D_RatiowrtMps[idx_2d] = (1.0*TempMatch3D2D.size())/vpObj3DMps.size();
                FrameObj2DIdx_Match3D2D_RatioIoU[idx_2d] = (1.0*TempMatch3D2D.size())/(vpObj3DMps.size()+vFrameKpIndices.size());
                ///////////////3D-2D MapPoint-KeyPoint Matches Check END/////////////


                ///////////////////Position Distance Check//////////////
                cv::Mat UnprojectedCenterW = cv::Mat::zeros(cv::Size(1,3), CV_32F);
                for(int i=0;i<vFrameKpIndices.size();i++)
                {
                    cv::Mat PosW = F.UnprojectStereo(vFrameKpIndices[i]);
                    UnprojectedCenterW = UnprojectedCenterW+PosW;
                }
                UnprojectedCenterW = UnprojectedCenterW/(1.0*vFrameKpIndices.size());

                //cout<<"Object2D "<<idx_2d<<" Unprojected CenterW = "<<UnprojectedCenterW<<endl;

                float minDist = FLT_MAX;
                float meanDist = 0.0;
                //cout<<"Object3D "<<pObj3D->mTrackID<<" Old CenterW = ";
                for(auto obs: vObs)
                {
                    cv::Mat OldCenterW = Converter::toCvMat(obs.ObjectCenterW);
                    float Dist = cv::norm(OldCenterW-UnprojectedCenterW);
                    meanDist = meanDist+Dist;
                    //cout<<OldCenterW<<", distance = "<<Dist<<".";
                    if(Dist < minDist)
                    {
                        minDist = Dist;
                    }
                }
                meanDist = meanDist/(1.0*vObs.size());
                //cout<<endl;
                FrameObj2DIdx_MeanDist[idx_2d] = meanDist;
                FrameObj2DIdx_MinDist[idx_2d] = minDist;

                //////////////////Position Distance Check END///////////////

                ///////////////////BBox3D Check//////////////



                ////////////BBox3D Check END/////////
                float MaxX = FLT_MIN;
                float MaxY = FLT_MIN;
                float MaxZ = FLT_MIN;
                float MinX = FLT_MAX;
                float MinY = FLT_MAX;
                float MinZ = FLT_MAX;

                for(int i=0;i<vFrameKpIndices.size();i++)
                {
                    cv::Mat PosW = F.UnprojectStereo(vFrameKpIndices[i]);
                    float Xw = PosW.at<float>(0);  
                    float Yw = PosW.at<float>(1);
                    float Zw = PosW.at<float>(2);

                    MaxX = max(MaxX, Xw);
                    MinX = min(MinX, Xw);
                    MaxY = max(MaxY, Yw);
                    MinY = min(MinY, Yw);
                    MaxZ = max(MaxZ, Zw);
                    MinZ = min(MinZ, Zw);
                }

                vector<float> FrameObj2DBBox3D = {MaxX, MaxY, MaxZ, MinX, MinY, MinZ};

                FrameObj2DIdx_BBox3DIoU[idx_2d] = Compute3DIoU(ObjBBox3D, FrameObj2DBBox3D);
                FrameObj2DIdx_BBox3DOverlapRatio[idx_2d] = Compute3DOverlapRatio(ObjBBox3D, FrameObj2DBBox3D);
            }


            /////////////////////////Analyze//////////////////////////////////
            //-------------Appearance Similarity Analyze-------------//
            //cout<<"*********Similarity Check Analyze*******"<<endl;
            float MaxSimilarity = 0.0;
            int MaxSimilarityFrameObj2DIdx = -1;
            if(!FrameObj2DIdx_Similarity.empty())
            {
                vector<pair<int,float>> vFrameObj2DIdx_Similarity = vector<pair<int,float>>(FrameObj2DIdx_Similarity.begin(), FrameObj2DIdx_Similarity.end());
                sort(vFrameObj2DIdx_Similarity.begin(), vFrameObj2DIdx_Similarity.end(), [&] (pair<int,float> a, pair<int,float> b) {return a.second<b.second;});
                MaxSimilarity = vFrameObj2DIdx_Similarity.back().second;
                MaxSimilarityFrameObj2DIdx = vFrameObj2DIdx_Similarity.back().first;
                // for(auto temp:vFrameObj2DIdx_Similarity)
                // {
                //     cout<<"Appearance Similarity: Object3D "<<pObj3D->mTrackID<<" <-> Object2D "<<temp.first<<" ="<<temp.second<<endl;
                // }
                // cout<<"Max Appearance Similarity: Object3D "<<pObj3D->mTrackID<<" <-> Object2D "<<MaxSimilarityFrameObj2DIdx<<" ="<<MaxSimilarity<<endl;
            }

            //--------END----------//


            //----------Position Distance Analyze---------//
            //cout<<"*********Position Distance Check Analyze*******"<<endl;
            float MinMinPosDist = 0.0;
            int MinMinPosDistFrameObj2DIdx = -1;
            if(!FrameObj2DIdx_MinDist.empty())
            {
                vector<pair<int,float>> vFrameObj2DIdx_MinDist = vector<pair<int,float>>(FrameObj2DIdx_MinDist.begin(), FrameObj2DIdx_MinDist.end());
                sort(vFrameObj2DIdx_MinDist.begin(), vFrameObj2DIdx_MinDist.end(), [&] (pair<int,float> a, pair<int,float> b) {return a.second<b.second;});
                MinMinPosDist = vFrameObj2DIdx_MinDist.front().second;
                MinMinPosDistFrameObj2DIdx = vFrameObj2DIdx_MinDist.front().first;
                // for(auto temp:vFrameObj2DIdx_MinDist)
                // {
                //     cout<<"Min Position Distance: Object3D "<<pObj3D->mTrackID<<" <-> Object2D "<<temp.first<<" ="<<temp.second<<endl;
                // }
                // cout<<"Min Min Position Distance: Object3D "<<pObj3D->mTrackID<<" <-> Object2D "<<MinMinPosDistFrameObj2DIdx<<" ="<<MinMinPosDist<<endl;
            }
            float MinMeanPosDist = FLT_MAX;
            int MinMeanPosDistFrameObj2DIdx = -1;
            if(!FrameObj2DIdx_MeanDist.empty())
            {
                vector<pair<int,float>> vFrameObj2DIdx_MeanDist = vector<pair<int,float>>(FrameObj2DIdx_MeanDist.begin(), FrameObj2DIdx_MeanDist.end());
                sort(vFrameObj2DIdx_MeanDist.begin(), vFrameObj2DIdx_MeanDist.end(), [&] (pair<int,float> a, pair<int,float> b) {return a.second<b.second;});
                MinMeanPosDist = vFrameObj2DIdx_MeanDist.front().second;
                MinMeanPosDistFrameObj2DIdx = vFrameObj2DIdx_MeanDist.front().first;
                // for(auto temp:vFrameObj2DIdx_MeanDist)
                // {
                //     cout<<"Mean Position Distance: Object3D "<<pObj3D->mTrackID<<" <-> Object2D "<<temp.first<<" ="<<temp.second<<endl;
                // }
                // cout<<"Min Mean Position Distance: Object3D "<<pObj3D->mTrackID<<" <-> Object2D "<<MinMeanPosDistFrameObj2DIdx<<" ="<<MinMeanPosDist<<endl;
            }
            //----------END-----------//


            //----------BBox3D Analyze---------//
            //cout<<"********BBox3D IoU Analyze*******"<<endl;
            // float MaxBBox3DIoU = 0.0;
            // int MaxBBox3DIoUFrameObj2DIdx = -1;
            // if(!FrameObj2DIdx_BBox3DIoU.empty())
            // {
            //     vector<pair<int,float>> vFrameObj2DIdx_BBox3DIoU = vector<pair<int, float>>(FrameObj2DIdx_BBox3DIoU.begin(), FrameObj2DIdx_BBox3DIoU.end());
            //     sort( vFrameObj2DIdx_BBox3DIoU.begin(),  vFrameObj2DIdx_BBox3DIoU.end(), [&] (pair<int, float> a, pair<int,float> b) {return a.second<b.second;});
            //     MaxBBox3DIoU = vFrameObj2DIdx_BBox3DIoU.back().second;
            //     MaxBBox3DIoUFrameObj2DIdx = vFrameObj2DIdx_BBox3DIoU.back().first;
            //     for(auto temp:vFrameObj2DIdx_BBox3DIoU)
            //     {
            //         cout<<"BBox3D IoU: Object3D "<<pObj3D->mTrackID<<" <-> Object2D "<<temp.first<<" = "<<temp.second<<endl;
            //     }
            //     cout<<"Max BBox3D IoU: Object3D "<<pObj3D->mTrackID<<" <-> Object2D "<<MaxBBox3DIoUFrameObj2DIdx <<" ="<<MaxBBox3DIoU<<endl;
            // }
            //cout<<"********BBox3D Overlap ratio Analyze*******"<<endl;
            float MaxBBox3DOverlapRatio = 0.0;
            int MaxBBox3DOverlapRatioFrameObj2DIdx = -1;
            if(!FrameObj2DIdx_BBox3DOverlapRatio.empty())
            {
                vector<pair<int,float>> vFrameObj2DIdx_BBox3DOverlapRatio = vector<pair<int, float>>(FrameObj2DIdx_BBox3DOverlapRatio.begin(), FrameObj2DIdx_BBox3DOverlapRatio.end());
                sort( vFrameObj2DIdx_BBox3DOverlapRatio.begin(),  vFrameObj2DIdx_BBox3DOverlapRatio.end(), [&] (pair<int, float> a, pair<int,float> b) {return a.second<b.second;});
                MaxBBox3DOverlapRatio = vFrameObj2DIdx_BBox3DOverlapRatio.back().second;
                MaxBBox3DOverlapRatioFrameObj2DIdx = vFrameObj2DIdx_BBox3DOverlapRatio.back().first;
                // for(auto temp:vFrameObj2DIdx_BBox3DOverlapRatio)
                // {
                //     cout<<"BBox3D IoU: Object3D "<<pObj3D->mTrackID<<" <-> Object2D "<<temp.first<<" = "<<temp.second<<endl;
                // }
                // cout<<"Max BBox3D IoU: Object3D "<<pObj3D->mTrackID<<" <-> Object2D "<<MaxBBox3DOverlapRatioFrameObj2DIdx <<" ="<<MaxBBox3DOverlapRatio<<endl;
            }

            //---------3D-2D Match Analyze--------//
            // cout<<"*********3D-2D Match Check Analyze*******"<<endl;
            // int MaxMatch3D2DNum = 0;
            // int MaxMatch3D2DFrameObj2DIdx = -1;
            // if(!FrameObj2DIdx_Match3D2D.empty())
            // {
            //     vector<pair<int,vector<pair<int,int>>>> vFrameObj2DIdx_Match3D2D = vector<pair<int, vector<pair<int,int>>>>(FrameObj2DIdx_Match3D2D.begin(), FrameObj2DIdx_Match3D2D.end());
            //     sort(vFrameObj2DIdx_Match3D2D.begin(), vFrameObj2DIdx_Match3D2D.end(), [&] (pair<int, vector<pair<int,int>>> a, pair<int, vector<pair<int,int>>> b) {return a.second.size()<b.second.size();});        
            //     int MaxMatch3D2DNum = vFrameObj2DIdx_Match3D2D.back().second.size();
            //     int MaxMatch3D2DFrameObj2DIdx = vFrameObj2DIdx_Match3D2D.back().first;
            //     for(auto temp:vFrameObj2DIdx_Match3D2D)
            //     {
            //         cout<<"3D-2D Match: Object3D "<<pObj3D->mTrackID<<" <-> Object2D "<<temp.first<<" ="<<temp.second.size()<<endl;
            //     }
            //     cout<<"Max 3D-2D Match: Object3D "<<pObj3D->mTrackID<<" <-> Object2D "<<MaxMatch3D2DFrameObj2DIdx<<" ="<<MaxMatch3D2DNum<<endl;             
            // }
            // float MaxMatch3D2DRatiowrtKps = 0.0;
            // int MaxMatch3D2DRatiowrtKpsFrameObj2DIdx = -1;
            // if(!FrameObj2DIdx_Match3D2D_RatiowrtKps.empty())
            // {
            //     vector<pair<int,float>> vFrameObj2DIdx_Match3D2D_RatiowrtKps = vector<pair<int, float>>(FrameObj2DIdx_Match3D2D_RatiowrtKps.begin(), FrameObj2DIdx_Match3D2D_RatiowrtKps.end());
            //     sort(vFrameObj2DIdx_Match3D2D_RatiowrtKps.begin(), vFrameObj2DIdx_Match3D2D_RatiowrtKps.end(), [&] (pair<int, float> a, pair<int,float> b) {return a.second<b.second;});
            //     MaxMatch3D2DRatiowrtKps = vFrameObj2DIdx_Match3D2D_RatiowrtKps.back().second;
            //     MaxMatch3D2DRatiowrtKpsFrameObj2DIdx = vFrameObj2DIdx_Match3D2D_RatiowrtKps.back().first;  
            //     for(auto temp:vFrameObj2DIdx_Match3D2D_RatiowrtKps)
            //     {
            //         cout<<"3D-2D Match ratio wrt Kps: Object3D "<<pObj3D->mTrackID<<" <-> Object2D "<<temp.first<<" ="<<temp.second<<endl;
            //     }
            //     cout<<"Max 3D-2D Match ratio wrt Kps: Object3D "<<pObj3D->mTrackID<<" <-> Object2D "<<MaxMatch3D2DRatiowrtKpsFrameObj2DIdx<<" ="<<MaxMatch3D2DRatiowrtKps<<endl;
            // }
            // float MaxMatch3D2DRatiowrtMps = 0.0;
            // int MaxMatch3D2DRatiowrtMpsFrameObj2DIdx = -1;
            // if(!FrameObj2DIdx_Match3D2D_RatiowrtMps.empty())
            // {
            //     vector<pair<int,float>> vFrameObj2DIdx_Match3D2D_RatiowrtMps = vector<pair<int, float>>(FrameObj2DIdx_Match3D2D_RatiowrtMps.begin(), FrameObj2DIdx_Match3D2D_RatiowrtMps.end());
            //     sort(vFrameObj2DIdx_Match3D2D_RatiowrtMps.begin(), vFrameObj2DIdx_Match3D2D_RatiowrtMps.end(), [&] (pair<int, float> a, pair<int,float> b) {return a.second<b.second;});
            //     MaxMatch3D2DRatiowrtMps = vFrameObj2DIdx_Match3D2D_RatiowrtMps.back().second;
            //     MaxMatch3D2DRatiowrtMpsFrameObj2DIdx = vFrameObj2DIdx_Match3D2D_RatiowrtMps.back().first;
            //     for(auto temp:vFrameObj2DIdx_Match3D2D_RatiowrtMps)
            //     {
            //         cout<<"3D-2D Match ratio wrt Mps: Object3D "<<pObj3D->mTrackID<<" <-> Object2D "<<temp.first<<" ="<<temp.second<<endl;
            //     }
            //     cout<<"Max 3D-2D Match ratio wrt Mps: Object3D "<<pObj3D->mTrackID<<" <-> Object2D "<<MaxMatch3D2DRatiowrtMpsFrameObj2DIdx<<" ="<<MaxMatch3D2DRatiowrtMps<<endl;
            // }
            // float MaxMatch3D2DRatioIoU= 0.0;
            // int MaxMatch3D2DRatioIoUFrameObj2DIdx = -1;
            // if(!FrameObj2DIdx_Match3D2D_RatioIoU.empty())
            // {
            //     vector<pair<int,float>> vFrameObj2DIdx_Match3D2D_RatioIoU = vector<pair<int, float>>(FrameObj2DIdx_Match3D2D_RatioIoU.begin(), FrameObj2DIdx_Match3D2D_RatioIoU.end());
            //     sort( vFrameObj2DIdx_Match3D2D_RatioIoU.begin(),  vFrameObj2DIdx_Match3D2D_RatioIoU.end(), [&] (pair<int, float> a, pair<int,float> b) {return a.second<b.second;});
            //     MaxMatch3D2DRatioIoU= vFrameObj2DIdx_Match3D2D_RatioIoU.back().second;
            //     MaxMatch3D2DRatioIoUFrameObj2DIdx = vFrameObj2DIdx_Match3D2D_RatioIoU.back().first;
            //     for(auto temp:vFrameObj2DIdx_Match3D2D_RatioIoU)
            //     {
            //         cout<<"3D-2D Match ratio IoU: Object3D "<<pObj3D->mTrackID<<" <-> Object2D "<<temp.first<<" ="<<temp.second<<endl;
            //     }
            //     cout<<"Max 3D-2D Match ratio IoU: Object3D "<<pObj3D->mTrackID<<" <-> Object2D "<<MaxMatch3D2DRatioIoUFrameObj2DIdx<<" ="<<MaxMatch3D2DRatioIoU<<endl;
            // }
            //------END--------//



            
            //---------------Ours---------------//
            //FrameObj2DIdx_MeanDist: 5.0 for outdoor scenes, 0.3 for indoor scenes
            if(MaxSimilarity > 0.8 && MaxSimilarityFrameObj2DIdx >= 0 && FrameObj2DIdx_MeanDist[MaxSimilarityFrameObj2DIdx]<0.3)
            {
                //cout<<"Assign Object3D "<<pObj3D->mTrackID<<" to Object2D "<<MaxSimilarityFrameObj2DIdx<<endl;
                F.mvpObject3Ds[MaxSimilarityFrameObj2DIdx] = pObj3D;
                F.mvObject2Ds[MaxSimilarityFrameObj2DIdx].track_id = pObj3D->mTrackID;
            }
            if(MinMinPosDist < 0.1 && MinMinPosDistFrameObj2DIdx >= 0 )
            {
                //cout<<"Assign Object3D "<<pObj3D->mTrackID<<" to Object2D "<<MinMinPosDistFrameObj2DIdx<<endl;
                F.mvpObject3Ds[MinMinPosDistFrameObj2DIdx] = pObj3D;
                F.mvObject2Ds[MinMinPosDistFrameObj2DIdx].track_id = pObj3D->mTrackID; 
            }

        }


    }

    
}

float ObjectMatcher::Compute3DIoU(Object3D *pObj3D1, Object3D *pObj3D2)
{
    vector<float> BBox3D1 = {pObj3D1->mMaxXw, pObj3D1->mMaxYw, pObj3D1->mMaxZw, pObj3D1->mMinXw, pObj3D1->mMinYw, pObj3D1->mMinZw};
    vector<float> BBox3D2 = {pObj3D2->mMaxXw, pObj3D2->mMaxYw, pObj3D2->mMaxZw, pObj3D2->mMinXw, pObj3D2->mMinYw, pObj3D2->mMinZw};

    return Compute3DIoU(BBox3D1, BBox3D2);
}

float ObjectMatcher::Compute3DIoU(vector<float> BBox3D1, vector<float> BBox3D2)
{
    float area1 = (BBox3D1[0]-BBox3D1[3])*(BBox3D1[1]-BBox3D1[4])*(BBox3D1[2]-BBox3D1[5]);
    float area2 = (BBox3D2[0]-BBox3D2[3])*(BBox3D2[1]-BBox3D2[4])*(BBox3D2[2]-BBox3D2[5]);

    float MinXw = max(BBox3D1[3], BBox3D2[3]);
    float MaxXw = min(BBox3D1[0], BBox3D2[0]);
    float MinYw = max(BBox3D1[4], BBox3D2[4]);
    float MaxYw = min(BBox3D1[1], BBox3D2[1]);
    float MinZw = max(BBox3D1[5], BBox3D2[5]);
    float MaxZw = min(BBox3D1[2], BBox3D2[2]);

    if(MinXw >= MaxXw || MinYw>= MaxYw || MinZw>=MaxZw)
    {
        return 0.0;
    }
    else
    {
        float inter_area = (MaxXw-MinXw)*(MaxYw-MinYw)*(MaxZw-MinZw);
        return inter_area/(area1+area2-inter_area);
    }
}


float ObjectMatcher::Compute3DOverlapRatio(Object3D *pObj3D1, Object3D *pObj3D2)
{
    vector<float> BBox3D1 = {pObj3D1->mMaxXw, pObj3D1->mMaxYw, pObj3D1->mMaxZw, pObj3D1->mMinXw, pObj3D1->mMinYw, pObj3D1->mMinZw};
    vector<float> BBox3D2 = {pObj3D2->mMaxXw, pObj3D2->mMaxYw, pObj3D2->mMaxZw, pObj3D2->mMinXw, pObj3D2->mMinYw, pObj3D2->mMinZw};

    return Compute3DOverlapRatio(BBox3D1, BBox3D2);
}

float ObjectMatcher::Compute3DOverlapRatio(vector<float> BBox3D1, vector<float> BBox3D2)
{
    //cout <<"BBox1: ";
    //for(auto temp: BBox3D1){ cout<<temp<<",";};cout<<endl;
    //cout <<"BBox2: ";
    //for(auto temp: BBox3D2){ cout<<temp<<",";};cout<<endl;
    float area1 = (BBox3D1[0]-BBox3D1[3])*(BBox3D1[1]-BBox3D1[4])*(BBox3D1[2]-BBox3D1[5]);
    float area2 = (BBox3D2[0]-BBox3D2[3])*(BBox3D2[1]-BBox3D2[4])*(BBox3D2[2]-BBox3D2[5]);

    float MinXw = max(BBox3D1[3], BBox3D2[3]);
    float MaxXw = min(BBox3D1[0], BBox3D2[0]);
    float MinYw = max(BBox3D1[4], BBox3D2[4]);
    float MaxYw = min(BBox3D1[1], BBox3D2[1]);
    float MinZw = max(BBox3D1[5], BBox3D2[5]);
    float MaxZw = min(BBox3D1[2], BBox3D2[2]);

    //cout<<"Intersection "<< MinXw <<", "<<MaxXw<<", "<<MinYw<<", "<<MaxYw<<", "<<MinZw<<", "<<MaxZw<<endl;


    if(MinXw >= MaxXw || MinYw>= MaxYw || MinZw>=MaxZw)
    {
        return 0.0;
    }
    else
    {
        float inter_area = (MaxXw-MinXw)*(MaxYw-MinYw)*(MaxZw-MinZw);
        return max(inter_area/area1, inter_area/area2);
    }

}

    

float ObjectMatcher::CalCosineSimilarity(cv::Mat &Hist1, cv::Mat &Hist2)
{
    float xy =0, xx=0, yy=0;
    for(int n=0;n<Hist1.cols;n++)
    {
        xy+=(Hist1.at<float>(0,n)*Hist2.at<float>(0,n));
        xx+=pow(Hist1.at<float>(0,n),2);
        yy+=pow(Hist2.at<float>(0,n),2);
    }
    
    float cosdist = xy/(sqrt(xx)*sqrt(yy));
    return cosdist;
    
}






void ObjectMatcher::RejectOutliersByRANSAC(vector<cv::Point2f> &pre_pts, vector<cv::Point2f> &cur_pts, vector<uchar> &status)
{
    vector<cv::Point2f> pre_good_pts, cur_good_pts;

    vector<int> good_indices;
    for(int i =0;i<status.size();i++)
    {
        if(status[i])
        {
            pre_good_pts.push_back(pre_pts[i]);
            cur_good_pts.push_back(cur_pts[i]);
            good_indices.push_back(i);
        }
    }

    if(pre_good_pts.size()==0)
    {
        return;
    }
    cv::Mat F = cv::findFundamentalMat(pre_good_pts,cur_good_pts, cv::FM_RANSAC, 0.1, 0.99);
    if(!F.empty())
    {
        for (int i = 0; i < pre_good_pts.size(); i++)
        {
            // Circle(pre_frame, pre_pts[i], 6, Scalar(255, 255, 0), 3);
            double A = F.at<double>(0, 0)*pre_good_pts[i].x + F.at<double>(0, 1)*pre_good_pts[i].y + F.at<double>(0, 2);
            double B = F.at<double>(1, 0)*pre_good_pts[i].x + F.at<double>(1, 1)*pre_good_pts[i].y + F.at<double>(1, 2);
            double C = F.at<double>(2, 0)*pre_good_pts[i].x + F.at<double>(2, 1)*pre_good_pts[i].y + F.at<double>(2, 2);
            double dd = fabs(A*cur_good_pts[i].x + B*cur_good_pts[i].y + C) / sqrt(A*A + B*B); //Epipolar constraints
            if (dd >= 0.1)
            {
                status[good_indices[i]] = 0;
            }
        }
    }
    
}

void ObjectMatcher::ReduceVector(vector<cv::Point2f> &vpts, vector<uchar> &status)
{
    int k=0;
    int N = vpts.size();
    for(int i=0;i<N;i++)
    {
        if(status[i])
        {
            vpts[k] = vpts[i];
            k++; 
        }
    }
    vpts.resize(k);
    
}

void ObjectMatcher::ReduceVector(vector<pair<int,int>> &matches, vector<uchar> &status)
{
    int k=0;
    int N = matches.size();
    for(int i=0;i<N;i++)
    {
        if(status[i])
        {
            matches[k] = matches[i];
            k++; 
        }
    }
    matches.resize(k);
    
}


int ObjectMatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}


void ObjectMatcher::ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
{
    int max1=0;
    int max2=0;
    int max3=0;

    for(int i=0; i<L; i++)
    {
        const int s = histo[i].size();
        if(s>max1)
        {
            max3=max2;
            max2=max1;
            max1=s;
            ind3=ind2;
            ind2=ind1;
            ind1=i;
        }
        else if(s>max2)
        {
            max3=max2;
            max2=s;
            ind3=ind2;
            ind2=i;
        }
        else if(s>max3)
        {
            max3=s;
            ind3=i;
        }
    }

    if(max2<0.1f*(float)max1)
    {
        ind2=-1;
        ind3=-1;
    }
    else if(max3<0.1f*(float)max1)
    {
        ind3=-1;
    }
}



}


