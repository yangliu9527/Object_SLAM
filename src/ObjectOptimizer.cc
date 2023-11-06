#include "Optimizer.h"
#include "ObjectOptimizer.h"

#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

#include<Eigen/StdVector>

#include "Converter.h"

#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>

#include<mutex>

namespace ORB_SLAM2
{
int N_AllSemanticConstraintNum = 0;


int ObjectOptimizer::PoseOptimization1(Frame *pFrame)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    int nInitialCorrespondences=0;

    // Set Frame vertex
    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    // Set MapPoint vertices
    const int N = pFrame->N;

    vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
    vector<size_t> vnIndexEdgeMono;
    vpEdgesMono.reserve(N);
    vnIndexEdgeMono.reserve(N);

    vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> vpEdgesStereo;
    vector<size_t> vnIndexEdgeStereo;
    vpEdgesStereo.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    const float deltaMono = sqrt(5.991);
    const float deltaStereo = sqrt(7.815);

    //----------For estalish semantic optimization-----------//
    //Common use
    vector<pcl::KdTreeFLANN<pcl::PointXY>*> vpMaskAreaKdTrees;
    vector<pcl::PointCloud<pcl::PointXY>::Ptr> vpMaskAreaClouds; 
    vector<int> vnMatchedObj3DIndices;
    int N_O = pFrame->N_O; 

    //For Initial optimization
    vector<MapPoint*> vpFrameMps = pFrame->mvpMapPoints;
    vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpSemanticEdgsInit;
    vector<pair<int,int>> vObjIndexMpIndexInit;
    vector<bool> vbOutliersInit;

    //For the consequent optimization
    vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpSemanticEdgsJoint;
    vector<pair<int,int>> vObjIndexMpIndexJoint;
    vector<bool> vbOutliersJoint;

    int nSemNum = 0;

    {


    //**********************Create the Semantic Constraints for the Whole Optimization*****************//
    for(int idx_obj=0;idx_obj<N_O;idx_obj++)
    {
        Object3D* pObj3D = pFrame->mvpObject3Ds[idx_obj];
        if(pObj3D)
        {
            vector<MapPoint*> vpObjMps = pObj3D->mvpMapPoints;
            int N_Mp = vpObjMps.size();
            vnMatchedObj3DIndices.push_back(idx_obj);
            //Prepare the ikd-tree for nearest search 
            Object2D Obj2D = pFrame->mvObject2Ds[idx_obj];
            cv::Mat mask = Obj2D.mask;
            pcl::PointCloud<pcl::PointXY>::Ptr mask_area(new pcl::PointCloud<pcl::PointXY>());
            for(int row=0;row<mask.rows;row++)
            {
                for(int col=0;col<mask.cols;col++)
                {
                    if((int)mask.ptr<uchar>(row)[col] == 255)
                    {
                        pcl::PointXY p;
                        p.x = col;
                        p.y = row;
                        mask_area->points.push_back(p);
                    }
                }
            }
            vpMaskAreaClouds.push_back(mask_area);
            pcl::KdTreeFLANN<pcl::PointXY>* kd_tree = new pcl::KdTreeFLANN <pcl::PointXY>();
            kd_tree->setInputCloud(mask_area);
            vpMaskAreaKdTrees.push_back(kd_tree);

            //------------Establish semantic constraints for the initial optimization--------//
            for(int idx_mp=0;idx_mp<vpFrameMps.size();idx_mp++)
            {
                MapPoint* pMp = vpFrameMps[idx_mp];
                if(pMp)
                {
                    auto it = find(vpObjMps.begin(), vpObjMps.end(), pMp);
                    if(it!=vpObjMps.end())
                    {
                        int Obj2DIndex = pFrame->mvObjectKpIndices[idx_mp].first;
                        if(Obj2DIndex<0 || Obj2DIndex!=idx_obj)
                        {
                            //The match belong to M_joint, establish the semantic constraint
                            cv::Mat PosW = pMp->GetWorldPos();
                            cv::Mat PosC = (pFrame->GetRcw())*PosW+(pFrame->Gettcw());
                            float x = PosC.at<float>(0,0)/PosC.at<float>(0,2);
                            float y = PosC.at<float>(0,1)/PosC.at<float>(0,2);
                            float u = pFrame->fx*x+pFrame->cx;
                            float v = pFrame->fy*y+pFrame->cy;
                            if(u>=pFrame->mnMinX && v>=pFrame->mnMinY && u<=pFrame->mnMaxX &&v <=pFrame->mnMaxY)
                            {
                                pcl::PointXY p;
                                p.x = u;
                                p.y = v;
                                vector<int> index;
                                vector<float> distance; 
                                
                                if(kd_tree->nearestKSearch(p, 1, index, distance))
                                {
                                    cout<<distance[0]<<endl;
                                    if(distance[0]<1.0)
                                        continue;
                                    pcl::PointXY obs_p = mask_area->points[index[0]];
                                    Eigen::Vector2d obs(obs_p.x, obs_p.y);
                                    g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();
                                    e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                                    e->setMeasurement(obs);
                                    const float invSigma2 = pFrame->mvInvLevelSigma2[0];
                                    e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);
                                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                                    e->setRobustKernel(rk);
                                    rk->setDelta(deltaMono);
                                    e->fx = pFrame->fx;
                                    e->fy = pFrame->fy;
                                    e->cx = pFrame->cx;
                                    e->cy = pFrame->cy;
                                    cv::Mat Xw = pMp->GetWorldPos();
                                    e->Xw[0] = Xw.at<float>(0);
                                    e->Xw[1] = Xw.at<float>(1);
                                    e->Xw[2] = Xw.at<float>(2);
                                    optimizer.addEdge(e);
                                    vpSemanticEdgsInit.push_back(e); 
                                    vObjIndexMpIndexInit.push_back(make_pair(vnMatchedObj3DIndices.size()-1,idx_mp));
                                    vbOutliersInit.push_back(false);
                                    nSemNum++;
                                    cout<<"add a new semantic constraint"<<endl;
                                }
                            }
                        }
                    }
                }

            }
            //---------------------END------------------//

            
            //-----------------Establish pure semantic constraints---------------//           
            // for(int idx_Mp=0;idx_Mp<N_Mp;idx_Mp++)
            // {   
            //     MapPoint* pMp = vpObjMps[idx_Mp];
            //     cv::Mat PosW = pMp->GetWorldPos();
            //     cv::Mat PosC = (pFrame->GetRcw())*PosW+(pFrame->Gettcw());
            //     float x = PosC.at<float>(0,0)/PosC.at<float>(0,2);
            //     float y = PosC.at<float>(0,1)/PosC.at<float>(0,2);
            //     float u = pFrame->fx*x+pFrame->cx;
            //     float v = pFrame->fy*y+pFrame->cy;
            //     if(u<0 || v<0 || u> mask.cols || v > mask.rows || (int)mask.ptr<uchar>(int(v))[int(u)] == 255)
            //         continue;
            //     pcl::PointXY p;
            //     p.x = u;
            //     p.y = v;
            //     vector<int> index;
            //     vector<float> distance; 
            //     kd_tree->nearestKSearch(p, 1, index, distance);
            //     if(!index.empty())
            //     {
            //         if(distance[0]<10)
            //         {
            //             pcl::PointXY obs_p = mask_area->points[index[0]];
            //             Eigen::Vector2d obs(obs_p.x, obs_p.y);
            //             g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();
            //             e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
            //             e->setMeasurement(obs);
            //             const float invSigma2 = pFrame->mvInvLevelSigma2[0];
            //             e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);
            //             g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            //             e->setRobustKernel(rk);
            //             rk->setDelta(deltaMono);
            //             e->fx = pFrame->fx;
            //             e->fy = pFrame->fy;
            //             e->cx = pFrame->cx;
            //             e->cy = pFrame->cy;
            //             cv::Mat Xw = pMp->GetWorldPos();
            //             e->Xw[0] = Xw.at<float>(0);
            //             e->Xw[1] = Xw.at<float>(1);
            //             e->Xw[2] = Xw.at<float>(2);
            //             //optimizer.addEdge(e);
            //             vObjIndexMpIndexJoint.push_back(make_pair(idx_obj,idx_Mp));
            //             vbInitOutliersJoint.push_back(false);
            //         }
            //     }
            // }
        }
        
    }
    //------------------------END------------------------//


    unique_lock<mutex> lock(MapPoint::mGlobalMutex);
    for(int i=0; i<N; i++)
    {
        MapPoint* pMP = pFrame->mvpMapPoints[i];
        if(pMP)
        {
            // Monocular observation
            if(pFrame->mvuRight[i]<0)
            {
                nInitialCorrespondences++;
                pFrame->mvbOutlier[i] = false;

                Eigen::Matrix<double,2,1> obs;
                const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                obs << kpUn.pt.x, kpUn.pt.y;

                g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                e->setMeasurement(obs);
                const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaMono);

                e->fx = pFrame->fx;
                e->fy = pFrame->fy;
                e->cx = pFrame->cx;
                e->cy = pFrame->cy;
                cv::Mat Xw = pMP->GetWorldPos();
                e->Xw[0] = Xw.at<float>(0);
                e->Xw[1] = Xw.at<float>(1);
                e->Xw[2] = Xw.at<float>(2);

                optimizer.addEdge(e);

                vpEdgesMono.push_back(e);
                vnIndexEdgeMono.push_back(i);
            }
            else  // Stereo observation
            {
                nInitialCorrespondences++;
                pFrame->mvbOutlier[i] = false;

                //SET EDGE
                Eigen::Matrix<double,3,1> obs;
                const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                const float &kp_ur = pFrame->mvuRight[i];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                e->setMeasurement(obs);
                const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                e->setInformation(Info);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaStereo);

                e->fx = pFrame->fx;
                e->fy = pFrame->fy;
                e->cx = pFrame->cx;
                e->cy = pFrame->cy;
                e->bf = pFrame->mbf;
                cv::Mat Xw = pMP->GetWorldPos();
                e->Xw[0] = Xw.at<float>(0);
                e->Xw[1] = Xw.at<float>(1);
                e->Xw[2] = Xw.at<float>(2);

                optimizer.addEdge(e);

                vpEdgesStereo.push_back(e);
                vnIndexEdgeStereo.push_back(i);
            }
        }
    }

    }


    if(nInitialCorrespondences<3)
        return 0;

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    const float chi2Stereo[4]={7.815,7.815,7.815, 7.815};
    const int its[4]={10,10,10,10};    

    int nBad=0;
    for(size_t it=0; it<4; it++)
    {
        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);
        //Check outliers of semantic constraints in the initial optimization and establish pure semantic constraints
        if(it==0)
        {
            cv::Mat InitialPose = Converter::toCvMat(vSE3->estimate());
            cv::Mat InitialRcw = InitialPose.rowRange(0,3).colRange(0,3);
            cv::Mat Initialtcw = InitialPose.rowRange(0,3).col(3);
            for(size_t i=0, iend= vpSemanticEdgsInit.size();i<iend;i++)
            {
                g2o::EdgeSE3ProjectXYZOnlyPose* e = vpSemanticEdgsInit[i];
                cv::Mat Pw = (cv::Mat_<float>(3,1) << e->Xw[0], e->Xw[1], e->Xw[2]);
                cv::Mat Pc = InitialRcw*Pw+Initialtcw;
                float x = Pc.at<float>(0,0)/Pc.at<float>(0,2);
                float y = Pc.at<float>(1,0)/Pc.at<float>(0,2);
                float u = pFrame->fx*x+pFrame->cx;
                float v = pFrame->fy*y+pFrame->cy;
                if(u<pFrame->mnMinX || v<pFrame->mnMinY || u> pFrame->mnMaxX ||v > pFrame->mnMaxY)
                {
                    vbOutliersInit[i] = true;
                    e->setLevel(1);
                    nSemNum--;
                }
                else
                {
                    pcl::PointXY p;
                    p.x = u;
                    p.y = v;
                    vector<int> index;
                    vector<float> distance; 
                    vpMaskAreaKdTrees[vObjIndexMpIndexInit[i].first]->nearestKSearch(p, 1, index, distance);
                    if(index.empty())
                    {
                        if(distance[0]<10)
                        {
                            e->setLevel(0);
                        }
                        else
                        {
                            e->setLevel(1);
                            vbOutliersInit[i] = true;
                            nSemNum--;
                        }
                    }
                }
            }
            //-----------------Establish pure semantic constraints---------------// 
            // for(int idx_obj=0;idx_obj<vnMatchedObj3DIndices.size();idx_obj++)
            // {
            //     Object3D* pObj3D = pFrame->mvpObject3Ds[vnMatchedObj3DIndices[idx_obj]];
            //     vector<MapPoint*> vpObjMps = pObj3D->mvpMapPoints;
            //     int N_Mp = vpObjMps.size();
            //     for(int idx_Mp=0;idx_Mp<N_Mp;idx_Mp++)
            //     {   
            //         MapPoint* pMp = vpObjMps[idx_Mp];
            //         cv::Mat PosW = pMp->GetWorldPos();
            //         cv::Mat PosC = (InitialRcw*PosW+Initialtcw);
            //         float x = PosC.at<float>(0,0)/PosC.at<float>(0,2);
            //         float y = PosC.at<float>(0,1)/PosC.at<float>(0,2);
            //         float u = pFrame->fx*x+pFrame->cx;
            //         float v = pFrame->fy*y+pFrame->cy;
            //         if(u<0 || v<0 || u> pFrame->mvObject2Ds[vnMatchedObj3DIndices[idx_obj]].mask.cols || v > pFrame->mvObject2Ds[vnMatchedObj3DIndices[idx_obj]].mask.rows || (int)pFrame->mvObject2Ds[vnMatchedObj3DIndices[idx_obj]].mask.ptr<uchar>(int(v))[int(u)] == 255)
            //             continue;                
            //         pcl::PointXY p;
            //         p.x = u;
            //         p.y = v;
            //         vector<int> index;
            //         vector<float> distance; 
            //         vpMaskAreaKdTrees[idx_obj]->nearestKSearch(p, 1, index, distance);
            //         if(!index.empty())
            //         {
            //             if(distance[0]<10)
            //             {
            //                 pcl::PointXY obs_p = vpMaskAreaClouds[idx_obj]->points[index[0]];
            //                 Eigen::Vector2d obs(obs_p.x, obs_p.y);
            //                 g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();
            //                 e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
            //                 e->setMeasurement(obs);
            //                 const float invSigma2 = pFrame->mvInvLevelSigma2[0];
            //                 e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);
            //                 g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            //                 e->setRobustKernel(rk);
            //                 rk->setDelta(deltaMono);
            //                 e->fx = pFrame->fx;
            //                 e->fy = pFrame->fy;
            //                 e->cx = pFrame->cx;
            //                 e->cy = pFrame->cy;
            //                 cv::Mat Xw = pMp->GetWorldPos();
            //                 e->Xw[0] = Xw.at<float>(0);
            //                 e->Xw[1] = Xw.at<float>(1);
            //                 e->Xw[2] = Xw.at<float>(2);
            //                 optimizer.addEdge(e);
            //                 vObjIndexMpIndexJoint.push_back(make_pair(idx_obj,idx_Mp));
            //                 vbOutliersJoint.push_back(false);
            //                 nSemNum++;
            //             }
            //         }
            //     }
            // }       
        }

        //Check outliers of semantic constraints
        if(it >=1)
        {
            //Check outliers of initial semantic constraints
            cv::Mat OptimizedPose = Converter::toCvMat(vSE3->estimate());
            cv::Mat OptimizedRcw = OptimizedPose.rowRange(0,3).colRange(0,3);
            cv::Mat Optimizedtcw = OptimizedPose.rowRange(0,3).col(3);
            for(size_t i=0, iend= vpSemanticEdgsInit.size();i<iend;i++)
            {
                g2o::EdgeSE3ProjectXYZOnlyPose* e = vpSemanticEdgsInit[i];
                cv::Mat Pw = (cv::Mat_<float>(3,1) << e->Xw[0], e->Xw[1], e->Xw[2]);
                cv::Mat Pc = OptimizedRcw*Pw+Optimizedtcw;
                float x = Pc.at<float>(0,0)/Pc.at<float>(0,2);
                float y = Pc.at<float>(1,0)/Pc.at<float>(0,2);
                float u = pFrame->fx*x+pFrame->cx;
                float v = pFrame->fy*y+pFrame->cy;
                if(u<pFrame->mnMinX || v<pFrame->mnMinY || u> pFrame->mnMaxX ||v > pFrame->mnMaxY)
                {
                    vbOutliersInit[i] = true;
                    if(e->level()==0)
                    {
                        e->setLevel(1);
                        nSemNum--;
                    }
                }
                else
                {
                    pcl::PointXY p;
                    p.x = u;
                    p.y = v;
                    vector<int> index;
                    vector<float> distance; 
                    vpMaskAreaKdTrees[vObjIndexMpIndexInit[i].first]->nearestKSearch(p, 1, index, distance);
                    if(index.empty())
                    {
                        if(distance[0]<10)
                        {
                            if(e->level()==1)
                            {
                                e->setLevel(0);
                                nSemNum++;
                            }                           
                        }
                        else
                        {
                            if(e->level()==0)
                            {
                                e->setLevel(1);
                                vbOutliersInit[i] = true;
                                nSemNum--;
                            }
                            
                        }
                    }
                }
            }

            //Check outliers of pure semantic constraints
            // for(size_t i=0, iend= vpSemanticEdgsJoint.size();i<iend;i++)
            // {
            //     g2o::EdgeSE3ProjectXYZOnlyPose* e =  vpSemanticEdgsJoint[i];
            //     cv::Mat Pw = (cv::Mat_<float>(3,1) << e->Xw[0], e->Xw[1], e->Xw[2]);
            //     cv::Mat Pc = OptimizedRcw*Pw+Optimizedtcw;
            //     float x = Pc.at<float>(0,0)/Pc.at<float>(0,2);
            //     float y = Pc.at<float>(1,0)/Pc.at<float>(0,2);
            //     float u = pFrame->fx*x+pFrame->cx;
            //     float v = pFrame->fy*y+pFrame->cy;
            //     if(u<0 || v<0 || u> pFrame->mvObject2Ds[vObjIndexMpIndexJoint[i].first].mask.cols ||v > pFrame->mvObject2Ds[vObjIndexMpIndexJoint[i].first].mask.rows)
            //     {
            //         if(e->level()==0)
            //         {
            //             vbOutliersJoint[i] = true;
            //             e->setLevel(1);
            //             nSemNum--;
            //         }                   
            //     }
            //     else
            //     {
            //         pcl::PointXY p;
            //         p.x = u;
            //         p.y = v;
            //         vector<int> index;
            //         vector<float> distance; 
            //         vpMaskAreaKdTrees[vObjIndexMpIndexJoint[i].first]->nearestKSearch(p, 1, index, distance);
            //         if(index.empty())
            //         {
            //             if(distance[0]<10)
            //             {
            //                 if(e->level()==1)
            //                 {
            //                     e->setLevel(0);
            //                     nSemNum++;
            //                 }      
            //             }
            //             else
            //             {
            //                 if(e->level()==0)
            //                 {
            //                     e->setLevel(1);
            //                     vbOutliersJoint[i] = true;
            //                     nSemNum--;
            //                 }                           
            //             }
            //         }
            //     }
            // }
        }
        //---------------------END-------------------//


        
        

        nBad=0;
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Mono[it])
            {                
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
        {
            g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Stereo[it])
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {                
                e->setLevel(0);
                pFrame->mvbOutlier[idx]=false;
            }

            if(it==2)
                e->setRobustKernel(0);
        }

      

        if(optimizer.edges().size()<10)
            break;
    } 


    cout<<"Pose Optimization use "<<nSemNum<<" Semantic constraints"<<endl;
    // Recover optimized pose and return number of inliers
    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    cv::Mat pose = Converter::toCvMat(SE3quat_recov);
    pFrame->SetPose(pose);
    return nInitialCorrespondences-nBad;
}



int ObjectOptimizer::PoseOptimization2(Frame *pFrame)//Two step optimization
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    int nInitialCorrespondences=0;

    // Set Frame vertex
    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    // Set MapPoint vertices
    const int N = pFrame->N;

    vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
    vector<size_t> vnIndexEdgeMono;
    vpEdgesMono.reserve(N);
    vnIndexEdgeMono.reserve(N);

    vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> vpEdgesStereo;
    vector<size_t> vnIndexEdgeStereo;
    vpEdgesStereo.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    const float deltaMono = sqrt(5.991);
    const float deltaStereo = sqrt(7.815);

    //----------For estalish semantic optimization-----------//
    //Common use
    vector<pcl::KdTreeFLANN<pcl::PointXY>*> vpMaskAreaKdTrees;
    vector<pcl::PointCloud<pcl::PointXY>::Ptr> vpMaskAreaClouds; 
    vector<int> vnMatchedObj3DIndices;
    int N_O = pFrame->N_O; 

    //For Initial optimization
    vector<MapPoint*> vpFrameMps = pFrame->mvpMapPoints;
    vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpSemanticEdgsInit;
    vector<pair<int,int>> vObjIndexMpIndexInit;
    vector<bool> vbOutliersInit;

    //For the consequent optimization
    vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpSemanticEdgsJoint;
    vector<pair<int,int>> vObjIndexMpIndexJoint;
    vector<bool> vbOutliersJoint;

    int nSemNum = 0;

    {


    //**********************Create the Semantic Constraints for the Initial Optimization*****************//
    for(int idx_obj=0;idx_obj<N_O;idx_obj++)
    {
        Object3D* pObj3D = pFrame->mvpObject3Ds[idx_obj];
        if(pObj3D)
        {
            vector<MapPoint*> vpObjMps = pObj3D->mvpMapPoints;
            int N_Mp = vpObjMps.size();
            vnMatchedObj3DIndices.push_back(idx_obj);
            //Prepare the ikd-tree for nearest search 
            Object2D Obj2D = pFrame->mvObject2Ds[idx_obj];
            cv::Mat mask = Obj2D.mask;
            pcl::PointCloud<pcl::PointXY>::Ptr mask_area(new pcl::PointCloud<pcl::PointXY>());
            for(int row=0;row<mask.rows;row++)
            {
                for(int col=0;col<mask.cols;col++)
                {
                    if((int)mask.ptr<uchar>(row)[col] == 255)
                    {
                        pcl::PointXY p;
                        p.x = col;
                        p.y = row;
                        mask_area->points.push_back(p);
                    }
                }
            }
            vpMaskAreaClouds.push_back(mask_area);
            pcl::KdTreeFLANN<pcl::PointXY>* kd_tree = new pcl::KdTreeFLANN <pcl::PointXY>();
            kd_tree->setInputCloud(mask_area);
            vpMaskAreaKdTrees.push_back(kd_tree);

            //------------Establish semantic constraints for the initial optimization--------//
            cout<<"Establishing M_joint semantic constraints..."<<endl;
            for(int idx_mp=0;idx_mp<vpFrameMps.size();idx_mp++)
            {
                MapPoint* pMp = vpFrameMps[idx_mp];
                if(pMp)
                {
                    auto it = find(vpObjMps.begin(), vpObjMps.end(), pMp);
                    if(it!=vpObjMps.end())
                    {
                        int Obj2DIndex = pFrame->mvObjectKpIndices[idx_mp].first;
                        if(Obj2DIndex!=idx_obj)
                        {
                            //The match belong to M_joint, establish the semantic constraint
                            cv::KeyPoint Kp = pFrame->mvKeysUn[idx_mp];
                            float u = Kp.pt.x;
                            float v = Kp.pt.y;
                            pcl::PointXY p;
                            p.x = u;
                            p.y = v;
                            vector<int> index;
                            vector<float> distance; 
                            if(kd_tree->nearestKSearch(p, 1, index, distance))
                            {
                                if(distance[0]<1.0)
                                    continue;
                                pcl::PointXY obs_p = mask_area->points[index[0]];
                                Eigen::Vector2d obs(obs_p.x, obs_p.y);
                                g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();
                                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                                e->setMeasurement(obs);
                                const float invSigma2 = pFrame->mvInvLevelSigma2[0];
                                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);
                                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                                e->setRobustKernel(rk);
                                rk->setDelta(deltaMono);
                                e->fx = pFrame->fx;
                                e->fy = pFrame->fy;
                                e->cx = pFrame->cx;
                                e->cy = pFrame->cy;
                                cv::Mat Xw = pMp->GetWorldPos();
                                e->Xw[0] = Xw.at<float>(0);
                                e->Xw[1] = Xw.at<float>(1);
                                e->Xw[2] = Xw.at<float>(2);
                                optimizer.addEdge(e);
                                vpSemanticEdgsInit.push_back(e); 
                                vObjIndexMpIndexInit.push_back(make_pair(vnMatchedObj3DIndices.size()-1,idx_mp));
                                vbOutliersInit.push_back(false);
                                nSemNum++;
                            }

                        }
                    }
                }

            }
            //---------------------END------------------//

            
            //-----------------Establish pure semantic constraints---------------//           
            // for(int idx_Mp=0;idx_Mp<N_Mp;idx_Mp++)
            // {   
            //     MapPoint* pMp = vpObjMps[idx_Mp];
            //     cv::Mat PosW = pMp->GetWorldPos();
            //     cv::Mat PosC = (pFrame->GetRcw())*PosW+(pFrame->Gettcw());
            //     float x = PosC.at<float>(0,0)/PosC.at<float>(0,2);
            //     float y = PosC.at<float>(0,1)/PosC.at<float>(0,2);
            //     float u = pFrame->fx*x+pFrame->cx;
            //     float v = pFrame->fy*y+pFrame->cy;
            //     if(u<0 || v<0 || u> mask.cols || v > mask.rows || (int)mask.ptr<uchar>(int(v))[int(u)] == 255)
            //         continue;
            //     pcl::PointXY p;
            //     p.x = u;
            //     p.y = v;
            //     vector<int> index;
            //     vector<float> distance; 
            //     kd_tree->nearestKSearch(p, 1, index, distance);
            //     if(!index.empty())
            //     {
            //         if(distance[0]<10)
            //         {
            //             pcl::PointXY obs_p = mask_area->points[index[0]];
            //             Eigen::Vector2d obs(obs_p.x, obs_p.y);
            //             g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();
            //             e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
            //             e->setMeasurement(obs);
            //             const float invSigma2 = pFrame->mvInvLevelSigma2[0];
            //             e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);
            //             g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            //             e->setRobustKernel(rk);
            //             rk->setDelta(deltaMono);
            //             e->fx = pFrame->fx;
            //             e->fy = pFrame->fy;
            //             e->cx = pFrame->cx;
            //             e->cy = pFrame->cy;
            //             cv::Mat Xw = pMp->GetWorldPos();
            //             e->Xw[0] = Xw.at<float>(0);
            //             e->Xw[1] = Xw.at<float>(1);
            //             e->Xw[2] = Xw.at<float>(2);
            //             //optimizer.addEdge(e);
            //             vObjIndexMpIndexJoint.push_back(make_pair(idx_obj,idx_Mp));
            //             vbInitOutliersJoint.push_back(false);
            //         }
            //     }
            // }
        }
        
    }
    //------------------------END------------------------//


    unique_lock<mutex> lock(MapPoint::mGlobalMutex);

    for(int i=0; i<N; i++)
    {
        MapPoint* pMP = pFrame->mvpMapPoints[i];
        if(pMP)
        {
            // Monocular observation
            if(pFrame->mvuRight[i]<0)
            {
                nInitialCorrespondences++;
                pFrame->mvbOutlier[i] = false;

                Eigen::Matrix<double,2,1> obs;
                const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                obs << kpUn.pt.x, kpUn.pt.y;

                g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                e->setMeasurement(obs);
                const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaMono);

                e->fx = pFrame->fx;
                e->fy = pFrame->fy;
                e->cx = pFrame->cx;
                e->cy = pFrame->cy;
                cv::Mat Xw = pMP->GetWorldPos();
                e->Xw[0] = Xw.at<float>(0);
                e->Xw[1] = Xw.at<float>(1);
                e->Xw[2] = Xw.at<float>(2);

                optimizer.addEdge(e);

                vpEdgesMono.push_back(e);
                vnIndexEdgeMono.push_back(i);
            }
            else  // Stereo observation
            {
                nInitialCorrespondences++;
                pFrame->mvbOutlier[i] = false;

                //SET EDGE
                Eigen::Matrix<double,3,1> obs;
                const cv::KeyPoint &kpUn = pFrame->mvKeysUn[i];
                const float &kp_ur = pFrame->mvuRight[i];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                e->setMeasurement(obs);
                const float invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                e->setInformation(Info);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaStereo);

                e->fx = pFrame->fx;
                e->fy = pFrame->fy;
                e->cx = pFrame->cx;
                e->cy = pFrame->cy;
                e->bf = pFrame->mbf;
                cv::Mat Xw = pMP->GetWorldPos();
                e->Xw[0] = Xw.at<float>(0);
                e->Xw[1] = Xw.at<float>(1);
                e->Xw[2] = Xw.at<float>(2);

                optimizer.addEdge(e);

                vpEdgesStereo.push_back(e);
                vnIndexEdgeStereo.push_back(i);
            }
        }
    }

    }


    if(nInitialCorrespondences<3)
        return 0;

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    const float chi2Stereo[4]={7.815,7.815,7.815, 7.815};
    const int its[4]={10,10,10,10};    

    int nBad=0;
    for(size_t it=0; it<4; it++)
    {
        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);
        //Check outliers of semantic constraints in the initial optimization and establish pure semantic constraints
        if(it==0)
        {
            
            cv::Mat InitialPose = Converter::toCvMat(vSE3->estimate());
            cv::Mat InitialRcw = InitialPose.rowRange(0,3).colRange(0,3);
            cv::Mat Initialtcw = InitialPose.rowRange(0,3).col(3);

            //Check semantic outliers in M_Joint
            for(size_t i=0, iend= vpSemanticEdgsInit.size();i<iend;i++)
            {
                g2o::EdgeSE3ProjectXYZOnlyPose* e = vpSemanticEdgsInit[i];
                cv::Mat Pw = (cv::Mat_<float>(3,1) << e->Xw[0], e->Xw[1], e->Xw[2]);
                cv::Mat Pc = InitialRcw*Pw+Initialtcw;
                float x = Pc.at<float>(0,0)/Pc.at<float>(0,2);
                float y = Pc.at<float>(1,0)/Pc.at<float>(0,2);
                float u = pFrame->fx*x+pFrame->cx;
                float v = pFrame->fy*y+pFrame->cy;
                if(u<pFrame->mnMinX || v<pFrame->mnMinY || u> pFrame->mnMaxX||v > pFrame->mnMaxY)
                {
                    vbOutliersInit[i] = true;
                    e->setLevel(1);
                    nSemNum--;
                }
                else
                {
                    pcl::PointXY p;
                    p.x = u;
                    p.y = v;
                    vector<int> index;
                    vector<float> distance; 
                    if(vpMaskAreaKdTrees[vObjIndexMpIndexInit[i].first]->nearestKSearch(p, 1, index, distance))
                    {
                        if(distance[0]>10)
                        {
                            e->setLevel(1);
                            vbOutliersInit[i] = true;
                            nSemNum--;
                        }
                        else
                        {
                            e->setLevel(0);
                            pcl::PointXY p_obs = vpMaskAreaClouds[vObjIndexMpIndexInit[i].first]->points[index[0]];
                            Eigen::Vector2d new_obs(p.x, p.y);
                            e->setMeasurement(new_obs);
                        }
                    }
                }
            }
           
            //-----------------Establish pure semantic constraints in M_semantic---------------// 
            for(int idx_obj=0;idx_obj<vnMatchedObj3DIndices.size();idx_obj++)
            {
                Object3D* pObj3D = pFrame->mvpObject3Ds[vnMatchedObj3DIndices[idx_obj]];
                vector<MapPoint*> vpObjMps = pObj3D->mvpMapPoints;
                int N_Mp = vpObjMps.size();
                for(int idx_Mp=0;idx_Mp<N_Mp;idx_Mp++)
                {   
                    MapPoint* pMp = vpObjMps[idx_Mp];
                    cv::Mat PosW = pMp->GetWorldPos();
                    cv::Mat PosC = (InitialRcw*PosW+Initialtcw);
                    float x = PosC.at<float>(0,0)/PosC.at<float>(0,2);
                    float y = PosC.at<float>(0,1)/PosC.at<float>(0,2);
                    float u = pFrame->fx*x+pFrame->cx;
                    float v = pFrame->fy*y+pFrame->cy;
                    // if(u<pFrame->mnMinX || v<pFrame->mnMinY || u> pFrame->mnMaxX || v > pFrame->mnMaxY)
                    // {
                    //     continue;      
                    // }          
                    pcl::PointXY p;
                    p.x = u;
                    p.y = v;
                    vector<int> index;
                    vector<float> distance;  
                            
                    if(vpMaskAreaKdTrees[idx_obj]->nearestKSearch(p, 1, index, distance))
                    {    
                        
                        if(distance[0]<10)
                        {
                            pcl::PointXY obs_p = vpMaskAreaClouds[idx_obj]->points[index[0]];
                            Eigen::Vector2d obs(obs_p.x, obs_p.y);
                            g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();
                            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                            e->setMeasurement(obs);
                            const float invSigma2 = pFrame->mvInvLevelSigma2[0];
                            e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);
                            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                            e->setRobustKernel(rk);
                            rk->setDelta(deltaMono);
                            e->fx = pFrame->fx;
                            e->fy = pFrame->fy;
                            e->cx = pFrame->cx;
                            e->cy = pFrame->cy;
                            cv::Mat Xw = pMp->GetWorldPos();
                            e->Xw[0] = Xw.at<float>(0);
                            e->Xw[1] = Xw.at<float>(1);
                            e->Xw[2] = Xw.at<float>(2);
                            optimizer.addEdge(e);
                            vObjIndexMpIndexJoint.push_back(make_pair(idx_obj,idx_Mp));
                            vbOutliersJoint.push_back(false);
                            nSemNum++;
                        }
                    }
                }
            }       
        }

        //Check outliers of semantic constraints
        if(it >=1)
        {
            cv::Mat OptimizedPose = Converter::toCvMat(vSE3->estimate());
            cv::Mat OptimizedRcw = OptimizedPose.rowRange(0,3).colRange(0,3);
            cv::Mat Optimizedtcw = OptimizedPose.rowRange(0,3).col(3);
            //Check outliers of initial semantic constraints
            for(size_t i=0, iend= vpSemanticEdgsInit.size();i<iend;i++)
            {
                g2o::EdgeSE3ProjectXYZOnlyPose* e = vpSemanticEdgsInit[i];
                cv::Mat Pw = (cv::Mat_<float>(3,1) << e->Xw[0], e->Xw[1], e->Xw[2]);
                cv::Mat Pc = OptimizedRcw*Pw+Optimizedtcw;
                float x = Pc.at<float>(0,0)/Pc.at<float>(0,2);
                float y = Pc.at<float>(1,0)/Pc.at<float>(0,2);
                float u = pFrame->fx*x+pFrame->cx;
                float v = pFrame->fy*y+pFrame->cy;
                if(u<pFrame->mnMinX || v<pFrame->mnMinY || u> pFrame->mnMaxX || v > pFrame->mnMaxY)
                {
                    e->setLevel(1);
                    if(vbOutliersInit[i])
                        continue;
                    else
                    {
                        vbOutliersInit[i] = true;
                        nSemNum--;
                    }
                }
                else
                {
                    pcl::PointXY p;
                    p.x = u;
                    p.y = v;
                    vector<int> index;
                    vector<float> distance; 
                    if(vpMaskAreaKdTrees[vObjIndexMpIndexInit[i].first]->nearestKSearch(p, 1, index, distance))
                    {
                        if(distance[0]>10)
                        {
                            e->setLevel(1);
                            if(vbOutliersInit[i])
                                continue;
                            else
                            {
                                vbOutliersInit[i] = true;
                                nSemNum--;
                            }
                        }
                        else
                        {
                            pcl::PointXY p_obs = vpMaskAreaClouds[vObjIndexMpIndexInit[i].first]->points[index[0]];
                            Eigen::Vector2d new_obs(p.x, p.y);
                            e->setMeasurement(new_obs);
                            e->setLevel(0);     
                            if(!vbOutliersInit[i])
                                continue;
                            else
                            {
                                vbOutliersInit[i] = false;
                                nSemNum++;
                            }
                        }
                    }
                }
            }


            //Check outliers of pure semantic constraints
            for(size_t i=0, iend= vpSemanticEdgsJoint.size();i<iend;i++)
            {
                g2o::EdgeSE3ProjectXYZOnlyPose* e =  vpSemanticEdgsJoint[i];
                cv::Mat Pw = (cv::Mat_<float>(3,1) << e->Xw[0], e->Xw[1], e->Xw[2]);
                cv::Mat Pc = OptimizedRcw*Pw+Optimizedtcw;
                float x = Pc.at<float>(0,0)/Pc.at<float>(0,2);
                float y = Pc.at<float>(1,0)/Pc.at<float>(0,2);
                float u = pFrame->fx*x+pFrame->cx;
                float v = pFrame->fy*y+pFrame->cy;
                if(u<0 || v<0 || u> pFrame->mvObject2Ds[vObjIndexMpIndexJoint[i].first].mask.cols ||v > pFrame->mvObject2Ds[vObjIndexMpIndexJoint[i].first].mask.rows)
                {
                    e->setLevel(1);
                    if(vbOutliersJoint[i])
                        continue;
                    else
                    {
                        vbOutliersJoint[i] = true;
                        nSemNum--;
                    }              
                }
                else
                {
                    pcl::PointXY p;
                    p.x = u;
                    p.y = v;
                    vector<int> index;
                    vector<float> distance; 
                    if(vpMaskAreaKdTrees[vObjIndexMpIndexJoint[i].first]->nearestKSearch(p, 1, index, distance))
                    {
                        if(distance[0]>10)
                        {
                            e->setLevel(1);
                            if(vbOutliersJoint[i])
                                continue;
                            else
                            {
                                vbOutliersJoint[i] = true;
                                nSemNum--;
                            }  
                        }
                        else
                        {
                            e->setLevel(0);
                            pcl::PointXY p_obs = vpMaskAreaClouds[vObjIndexMpIndexJoint[i].first]->points[index[0]];
                            Eigen::Vector2d new_obs(p.x, p.y);
                            e->setMeasurement(new_obs); 
                            if(!vbOutliersJoint[i])
                                continue;
                            else
                            {
                                vbOutliersJoint[i] = false;
                                nSemNum++;
                            }
                        }
                    }
                }
            }
        }
        //---------------------END-------------------//


        
        

        nBad=0;
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Mono[it])
            {                
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
                pFrame->mvbOutlier[idx]=false;
                e->setLevel(0);
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
        {
            g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if(pFrame->mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Stereo[it])
            {
                pFrame->mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {                
                e->setLevel(0);
                pFrame->mvbOutlier[idx]=false;
            }

            if(it==2)
                e->setRobustKernel(0);
        }

      

        if(optimizer.edges().size()<10)
            break;
    } 


    cout<<"Pose Optimization use "<<nSemNum<<" Semantic constraints"<<endl;
    N_AllSemanticConstraintNum = N_AllSemanticConstraintNum+nSemNum;
    // Recover optimized pose and return number of inliers
    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    cv::Mat pose = Converter::toCvMat(SE3quat_recov);
    pFrame->SetPose(pose);
    return nInitialCorrespondences-nBad;
}






}
