/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "MapDrawer.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include <pangolin/pangolin.h>
#include <mutex>

namespace ORB_SLAM2
{


MapDrawer::MapDrawer(Map* pMap, const string &strSettingPath):mpMap(pMap)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    mKeyFrameSize = fSettings["Viewer.KeyFrameSize"];
    mKeyFrameLineWidth = fSettings["Viewer.KeyFrameLineWidth"];
    mGraphLineWidth = fSettings["Viewer.GraphLineWidth"];
    mPointSize = fSettings["Viewer.PointSize"];
    mCameraSize = fSettings["Viewer.CameraSize"];
    mCameraLineWidth = fSettings["Viewer.CameraLineWidth"];

}

void MapDrawer::DrawMapPoints()
{
    const vector<MapPoint*> &vpMPs = mpMap->GetAllMapPoints();
    const vector<MapPoint*> &vpRefMPs = mpMap->GetReferenceMapPoints();

    set<MapPoint*> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());

    if(vpMPs.empty())
        return;

    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(0.0,0.0,0.0);

    for(size_t i=0, iend=vpMPs.size(); i<iend;i++)
    {
        if(vpMPs[i]->isBad() || spRefMPs.count(vpMPs[i]))
            continue;
        cv::Mat pos = vpMPs[i]->GetWorldPos();
        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
    }
    glEnd();

    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(1.0,0.0,0.0);

    for(set<MapPoint*>::iterator sit=spRefMPs.begin(), send=spRefMPs.end(); sit!=send; sit++)
    {
        if((*sit)->isBad())
            continue;
        cv::Mat pos = (*sit)->GetWorldPos();
        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));

    }

    glEnd();
}

void MapDrawer::DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph)
{
    const float &w = mKeyFrameSize;
    const float h = w*0.75;
    const float z = w*0.6;

    const vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();

    if(bDrawKF)
    {
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            KeyFrame* pKF = vpKFs[i];
            cv::Mat Twc = pKF->GetPoseInverse().t();

            glPushMatrix();

            glMultMatrixf(Twc.ptr<GLfloat>(0));

            glLineWidth(mKeyFrameLineWidth);
            glColor3f(0.0f,0.0f,1.0f);
            glBegin(GL_LINES);
            glVertex3f(0,0,0);
            glVertex3f(w,h,z);
            glVertex3f(0,0,0);
            glVertex3f(w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,h,z);

            glVertex3f(w,h,z);
            glVertex3f(w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(-w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(w,h,z);

            glVertex3f(-w,-h,z);
            glVertex3f(w,-h,z);
            glEnd();

            glPopMatrix();
        }
    }

    if(bDrawGraph)
    {
        glLineWidth(mGraphLineWidth);
        glColor4f(0.0f,1.0f,0.0f,0.6f);
        glBegin(GL_LINES);

        for(size_t i=0; i<vpKFs.size(); i++)
        {
            // Covisibility Graph
            const vector<KeyFrame*> vCovKFs = vpKFs[i]->GetCovisiblesByWeight(100);
            cv::Mat Ow = vpKFs[i]->GetCameraCenter();
            if(!vCovKFs.empty())
            {
                for(vector<KeyFrame*>::const_iterator vit=vCovKFs.begin(), vend=vCovKFs.end(); vit!=vend; vit++)
                {
                    if((*vit)->mnId<vpKFs[i]->mnId)
                        continue;
                    cv::Mat Ow2 = (*vit)->GetCameraCenter();
                    glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                    glVertex3f(Ow2.at<float>(0),Ow2.at<float>(1),Ow2.at<float>(2));
                }
            }

            // Spanning tree
            KeyFrame* pParent = vpKFs[i]->GetParent();
            if(pParent)
            {
                cv::Mat Owp = pParent->GetCameraCenter();
                glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                glVertex3f(Owp.at<float>(0),Owp.at<float>(1),Owp.at<float>(2));
            }

            // Loops
            set<KeyFrame*> sLoopKFs = vpKFs[i]->GetLoopEdges();
            for(set<KeyFrame*>::iterator sit=sLoopKFs.begin(), send=sLoopKFs.end(); sit!=send; sit++)
            {
                if((*sit)->mnId<vpKFs[i]->mnId)
                    continue;
                cv::Mat Owl = (*sit)->GetCameraCenter();
                glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                glVertex3f(Owl.at<float>(0),Owl.at<float>(1),Owl.at<float>(2));
            }
        }

        glEnd();
    }
}

/////////new codes/////////
void MapDrawer::DrawCurrentObjects()
{
    int N_Obj3D = mCurrentFrame.mvpObject3Ds.size();
    vector<Object3D*> vpObj3Ds = mCurrentFrame.mvpObject3Ds;
    cv::RNG rng;
    for(int i=0;i<N_Obj3D;i++)
    {
        Object3D* pObj3D = vpObj3Ds[i];
        if(pObj3D)
        {
            int TrackID = pObj3D->mTrackID;
            cv::Scalar Color = cv::Scalar(rng.uniform(0.0,1.0), rng.uniform(0.0,1.0), rng.uniform(0.0,1.0));
            vector<MapPoint*> vpMps = pObj3D->mvpMapPoints;
            Eigen::Vector3f CenterW = pObj3D->mCenterW;
            float MaxXw = pObj3D->mMaxXw;
            float MinXw = pObj3D->mMinXw;
            float MaxYw = pObj3D->mMaxYw;
            float MinYw = pObj3D->mMinYw;
            float MaxZw = pObj3D->mMaxZw;
            float MinZw = pObj3D->mMinZw;
            
            glPointSize(10);
            glBegin(GL_POINTS);
            glColor3f(1.0, 0, 0);
            glVertex3f(CenterW[0],CenterW[1],CenterW[2]);
            glEnd();
            for(int j=0;j<vpMps.size();j++)
            {
                cv::Mat PosW = vpMps[j]->GetWorldPos();
                glPointSize(7);
                glBegin(GL_POINTS);
                glColor3f(Color[0],Color[1],Color[2]);
                glVertex3f(PosW.at<float>(0),PosW.at<float>(1),PosW.at<float>(2));
                glEnd();
            }


            glLineWidth(3);
            glBegin(GL_LINES);
            glColor3f(Color[0],Color[1],Color[2]);
            
            Eigen::Vector3f p1 = Eigen::Vector3f(MaxXw,MaxYw,MaxZw);
            Eigen::Vector3f p2 = Eigen::Vector3f(MaxXw,MaxYw,MinZw);
            Eigen::Vector3f p3 = Eigen::Vector3f(MaxXw,MinYw,MaxZw);
            Eigen::Vector3f p4 = Eigen::Vector3f(MinXw,MaxYw,MaxZw);
            Eigen::Vector3f p5 = Eigen::Vector3f(MinXw,MinYw,MaxZw);
            Eigen::Vector3f p6 = Eigen::Vector3f(MinXw,MinYw,MinZw);
            Eigen::Vector3f p7 = Eigen::Vector3f(MinXw,MaxYw,MinZw);
            Eigen::Vector3f p8 = Eigen::Vector3f(MaxXw,MinYw,MinZw);
            

            glVertex3f(p8[0],p8[1],p8[2]);glVertex3f(p2[0],p2[1],p2[2]);
            glVertex3f(p2[0],p2[1],p2[2]);glVertex3f(p7[0],p7[1],p7[2]);
            glVertex3f(p7[0],p7[1],p7[2]);glVertex3f(p6[0],p6[1],p6[2]);
            glVertex3f(p6[0],p6[1],p6[2]);glVertex3f(p8[0],p8[1],p8[2]);
            glVertex3f(p8[0],p8[1],p8[2]);glVertex3f(p3[0],p3[1],p3[2]);
            glVertex3f(p2[0],p2[1],p2[2]);glVertex3f(p1[0],p1[1],p1[2]);
            glVertex3f(p7[0],p7[1],p7[2]);glVertex3f(p4[0],p4[1],p4[2]);
            glVertex3f(p6[0],p6[1],p6[2]);glVertex3f(p5[0],p5[1],p5[2]);
            glVertex3f(p5[0],p5[1],p5[2]);glVertex3f(p3[0],p3[1],p3[2]);
            glVertex3f(p3[0],p3[1],p3[2]);glVertex3f(p1[0],p1[1],p1[2]);
            glVertex3f(p1[0],p1[1],p1[2]);glVertex3f(p4[0],p4[1],p4[2]);
            glVertex3f(p4[0],p4[1],p4[2]);glVertex3f(p5[0],p5[1],p5[2]);


            glEnd();

        }
    }
}

void MapDrawer::DrawMapObjects(bool bDrawCube, bool bDrawQuadric, bool bDrawMapPoints, bool bDrawCenter)
{
    int N_Obj3D = mvpMapObj3Ds.size();
    vector<Object3D*> vpObj3Ds = mvpMapObj3Ds;
    cv::RNG rng;
    for(int i=0;i<N_Obj3D;i++)
    {
        Object3D* pObj3D = vpObj3Ds[i];
        if(pObj3D)
        {
            int TrackID = pObj3D->mTrackID;
            cv::Scalar Color = cv::Scalar(rng.uniform(0.0,1.0), rng.uniform(0.0,1.0), rng.uniform(0.0,1.0));
            Eigen::Vector3f CenterW = pObj3D->mCenterW;
            //Draw Center
            if(bDrawCenter)
            {
                glPointSize(10);
                glBegin(GL_POINTS);
                glColor3f(1.0, 0, 0);
                glVertex3f(CenterW[0],CenterW[1],CenterW[2]);
                glEnd();
            }



            //Draw MapPoints
            if(bDrawMapPoints)
            {
                vector<MapPoint*> vpMps = pObj3D->mvpMapPoints;
                for(int j=0;j<vpMps.size();j++)
                {
                    cv::Mat PosW = vpMps[j]->GetWorldPos();
                    glPointSize(7);
                    glBegin(GL_POINTS);
                    glColor3f(Color[0],Color[1],Color[2]);
                    glVertex3f(PosW.at<float>(0),PosW.at<float>(1),PosW.at<float>(2));
                    glEnd();
                }
            }
            


            float MaxXw = pObj3D->mMaxXw;
            float MinXw = pObj3D->mMinXw;
            float MaxYw = pObj3D->mMaxYw;
            float MinYw = pObj3D->mMinYw;
            float MaxZw = pObj3D->mMaxZw;
            float MinZw = pObj3D->mMinZw;

            

            //Draw Cube
            if(bDrawCube)
            {
                glLineWidth(3);
                glBegin(GL_LINES);
                glColor3f(Color[0],Color[1],Color[2]);
                Eigen::Vector3f p1 = Eigen::Vector3f(MaxXw,MaxYw,MaxZw);
                Eigen::Vector3f p2 = Eigen::Vector3f(MaxXw,MaxYw,MinZw);
                Eigen::Vector3f p3 = Eigen::Vector3f(MaxXw,MinYw,MaxZw);
                Eigen::Vector3f p4 = Eigen::Vector3f(MinXw,MaxYw,MaxZw);
                Eigen::Vector3f p5 = Eigen::Vector3f(MinXw,MinYw,MaxZw);
                Eigen::Vector3f p6 = Eigen::Vector3f(MinXw,MinYw,MinZw);
                Eigen::Vector3f p7 = Eigen::Vector3f(MinXw,MaxYw,MinZw);
                Eigen::Vector3f p8 = Eigen::Vector3f(MaxXw,MinYw,MinZw);
                glVertex3f(p8[0],p8[1],p8[2]);glVertex3f(p2[0],p2[1],p2[2]);
                glVertex3f(p2[0],p2[1],p2[2]);glVertex3f(p7[0],p7[1],p7[2]);
                glVertex3f(p7[0],p7[1],p7[2]);glVertex3f(p6[0],p6[1],p6[2]);
                glVertex3f(p6[0],p6[1],p6[2]);glVertex3f(p8[0],p8[1],p8[2]);
                glVertex3f(p8[0],p8[1],p8[2]);glVertex3f(p3[0],p3[1],p3[2]);
                glVertex3f(p2[0],p2[1],p2[2]);glVertex3f(p1[0],p1[1],p1[2]);
                glVertex3f(p7[0],p7[1],p7[2]);glVertex3f(p4[0],p4[1],p4[2]);
                glVertex3f(p6[0],p6[1],p6[2]);glVertex3f(p5[0],p5[1],p5[2]);
                glVertex3f(p5[0],p5[1],p5[2]);glVertex3f(p3[0],p3[1],p3[2]);
                glVertex3f(p3[0],p3[1],p3[2]);glVertex3f(p1[0],p1[1],p1[2]);
                glVertex3f(p1[0],p1[1],p1[2]);glVertex3f(p4[0],p4[1],p4[2]);
                glVertex3f(p4[0],p4[1],p4[2]);glVertex3f(p5[0],p5[1],p5[2]);
                glEnd();
            }
            

            //Draw Quadric
            if(bDrawQuadric)
            {
                GLUquadricObj *pObj = gluNewQuadric();
                cv::Mat ObjectPose = cv::Mat::eye(4,4,CV_32F);
                ObjectPose.at<float>(0,3) = CenterW[0];
                ObjectPose.at<float>(1,3) = CenterW[1];
                ObjectPose.at<float>(2,3) = CenterW[2];
                ObjectPose = ObjectPose.t();
                glPushMatrix();
                gluNewQuadric();
                glMultMatrixf(ObjectPose.ptr<GLfloat >(0));
                glScalef((GLfloat)(MaxXw-MinXw)/2, (GLfloat)(MaxYw-MinYw)/2, (GLfloat)(MaxZw-MinZw)/2);
                gluQuadricDrawStyle(pObj, GLU_LINE);
                gluQuadricNormals(pObj, GLU_NONE);
                glBegin(GL_COMPILE);
                glColor3f(Color[0],Color[1],Color[2]);
                gluSphere(pObj, 1., 15, 10);
                glEnd();
                glPopMatrix();
            }
            
        }
    }
}

void MapDrawer::Update(Tracking *pTracker)
{
    mCurrentFrame = pTracker->mCurrentFrame;
    mvpMapObj3Ds = mpMap->GetAllObject3Ds();
}

void MapDrawer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc)
{
    const float &w = mCameraSize;
    const float h = w*0.75;
    const float z = w*0.6;

    glPushMatrix();

#ifdef HAVE_GLES
        glMultMatrixf(Twc.m);
#else
        glMultMatrixd(Twc.m);
#endif

    glLineWidth(mCameraLineWidth);
    glColor3f(0.0f,1.0f,0.0f);
    glBegin(GL_LINES);
    glVertex3f(0,0,0);
    glVertex3f(w,h,z);
    glVertex3f(0,0,0);
    glVertex3f(w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,h,z);

    glVertex3f(w,h,z);
    glVertex3f(w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(-w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);

    glVertex3f(-w,-h,z);
    glVertex3f(w,-h,z);
    glEnd();

    glPopMatrix();
}


void MapDrawer::SetCurrentCameraPose(const cv::Mat &Tcw)
{
    unique_lock<mutex> lock(mMutexCamera);
    mCameraPose = Tcw.clone();
}

void MapDrawer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M)
{
    if(!mCameraPose.empty())
    {
        cv::Mat Rwc(3,3,CV_32F);
        cv::Mat twc(3,1,CV_32F);
        {
            unique_lock<mutex> lock(mMutexCamera);
            Rwc = mCameraPose.rowRange(0,3).colRange(0,3).t();
            twc = -Rwc*mCameraPose.rowRange(0,3).col(3);
        }

        M.m[0] = Rwc.at<float>(0,0);
        M.m[1] = Rwc.at<float>(1,0);
        M.m[2] = Rwc.at<float>(2,0);
        M.m[3]  = 0.0;

        M.m[4] = Rwc.at<float>(0,1);
        M.m[5] = Rwc.at<float>(1,1);
        M.m[6] = Rwc.at<float>(2,1);
        M.m[7]  = 0.0;

        M.m[8] = Rwc.at<float>(0,2);
        M.m[9] = Rwc.at<float>(1,2);
        M.m[10] = Rwc.at<float>(2,2);
        M.m[11]  = 0.0;

        M.m[12] = twc.at<float>(0);
        M.m[13] = twc.at<float>(1);
        M.m[14] = twc.at<float>(2);
        M.m[15]  = 1.0;
    }
    else
        M.SetIdentity();
}

} //namespace ORB_SLAM
