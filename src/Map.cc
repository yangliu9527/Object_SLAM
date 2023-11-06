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

#include "Map.h"
#include "ObjectMatcher.h"

#include<mutex>

namespace ORB_SLAM2
{

Map::Map():mnMaxKFid(0),mnBigChangeIdx(0)
{
}

///////////new codes/////////
//////////////////new codes//////////////////////
void Map::AddObject3D(Object3D* pObject3D)
{
    //unique_lock<mutex> lock(mMutexMap);
    mspObject3Ds.insert(pObject3D);
}

std::vector<Object3D*> Map::GetAllObject3Ds()
{
    //unique_lock<mutex> lock(mMutexMap);
    return vector<Object3D*>(mspObject3Ds.begin(),mspObject3Ds.end());
}

bool Map::CheckIfMerge(Object3D* pObj3D1, Object3D* pObj3D2)
{
    if(pObj3D1->mLabel!=pObj3D2->mLabel)
    {
        return false;
    }

    if(pObj3D1->mTrackID == pObj3D2->mTrackID)
    {
        return false;
    }


    float OverlapRatio = ObjectMatcher::Compute3DOverlapRatio(pObj3D1, pObj3D2);

    //cout<<"Object3D "<<pObj3D1->mTrackID<<" <-> pObject3D2 "<<pObj3D2->mTrackID<<" overlap ratio = "<<OverlapRatio<<endl;

    return OverlapRatio > 0.4;
}

void Map::ObjectMapRegularization()
{
    vector<Object3D*> vpObj3Ds = GetAllObject3Ds();
    vector<set<int>> vMergeSets;
    int N = vpObj3Ds.size();
    for(int idx1=0; idx1<N; idx1++)
    {
        Object3D* pObj3D1 = vpObj3Ds[idx1];
        for(int idx2=idx1+1; idx2<N; idx2++)
        {
            Object3D* pObj3D2 = vpObj3Ds[idx2];
            bool merge_flag = CheckIfMerge(pObj3D1, pObj3D2);
            if(merge_flag)
            {
                bool insert_flag = false;
                for(auto &merge_set: vMergeSets)
                {
                    if(merge_set.count(idx1) || merge_set.count(idx2))
                    {
                        merge_set.insert(idx1);
                        merge_set.insert(idx2);
                        insert_flag = true;
                    }
                }
                
                if(!insert_flag)
                {
                    set<int> new_set = {idx1, idx2};
                    vMergeSets.push_back(new_set);
                }

            }
        }
    }

    for(auto merge_set: vMergeSets)
    {
        vector<Object3D*> vpObj3DtoMerge;
        for(auto it=merge_set.begin(); it!=merge_set.end();it++)
        {
            vpObj3DtoMerge.push_back(vpObj3Ds[*it]);
        }

        MergeObject(vpObj3DtoMerge);
    }
}

void Map::MergeObject(vector<Object3D*> vpObj3DtoMerge)
{
    int MaxTrackID = -1;
    int MaxTrackIDindex;
    vector<MapPoint*> vpNewMapPoints;
    vector<ObjectObservation> vNewObs;
    //cout<<"Merge Object ";
    for(int idx=0; idx<vpObj3DtoMerge.size(); idx++)
    {
        Object3D* pObj3D = vpObj3DtoMerge[idx];
        //cout<<pObj3D->mTrackID<<", ";
        if(pObj3D->mTrackID>MaxTrackID)
        {
           MaxTrackIDindex = idx; 
           MaxTrackID = pObj3D->mTrackID;
        }        
        vpNewMapPoints.insert(vpNewMapPoints.end(), pObj3D->mvpMapPoints.begin(), pObj3D->mvpMapPoints.end());
        vNewObs.insert(vNewObs.end(), pObj3D->mvObservations.begin(), pObj3D->mvObservations.end());
    }
    //cout<<"..."<<endl;

    Object3D* pTargetObj3D = vpObj3DtoMerge[MaxTrackIDindex];
    pTargetObj3D->mvpMapPoints.swap(vpNewMapPoints);
    pTargetObj3D->mvObservations.swap(vNewObs);
    if(abs((int)pTargetObj3D->mvpMapPoints.size()-(int)vpNewMapPoints.size())>50)
        pTargetObj3D->RejectOutliers();
    pTargetObj3D->CalculateCenterAndSize();
    pTargetObj3D->mUpdateCnt++;
    for(int idx=0; idx<vpObj3DtoMerge.size(); idx++)
    {
        Object3D *pObj3D = vpObj3DtoMerge[idx];
        if(idx==MaxTrackIDindex)
            continue;
        else
        {
            pObj3D->mpReplaced = pTargetObj3D;
            //cout<<"Erase Object3D "<<pObj3D->mTrackID<<endl;
            if(mspObject3Ds.count(pObj3D));
                mspObject3Ds.erase(pObj3D);
        }
    }


}

void Map::ClearInvalidObject()
{
    vector<Object3D*> vpObj3Ds = GetAllObject3Ds();
    for(auto pObj3D: vpObj3Ds)
    {
        vector<MapPoint*> vpMps = pObj3D->mvpMapPoints;
        if(vpMps.size()<5)
        {
            mspObject3Ds.erase(pObj3D);
        }
    }
}

////////////////////////////////

void Map::AddKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexMap);
    mspKeyFrames.insert(pKF);
    if(pKF->mnId>mnMaxKFid)
        mnMaxKFid=pKF->mnId;
}

void Map::AddMapPoint(MapPoint *pMP)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapPoints.insert(pMP);
}

void Map::EraseMapPoint(MapPoint *pMP)
{
    unique_lock<mutex> lock(mMutexMap);
    mspMapPoints.erase(pMP);

    // TODO: This only erase the pointer.
    // Delete the MapPoint
}

void Map::EraseKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexMap);
    mspKeyFrames.erase(pKF);

    // TODO: This only erase the pointer.
    // Delete the MapPoint
}

void Map::SetReferenceMapPoints(const vector<MapPoint *> &vpMPs)
{
    unique_lock<mutex> lock(mMutexMap);
    mvpReferenceMapPoints = vpMPs;
}

void Map::InformNewBigChange()
{
    unique_lock<mutex> lock(mMutexMap);
    mnBigChangeIdx++;
}

int Map::GetLastBigChangeIdx()
{
    unique_lock<mutex> lock(mMutexMap);
    return mnBigChangeIdx;
}

vector<KeyFrame*> Map::GetAllKeyFrames()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<KeyFrame*>(mspKeyFrames.begin(),mspKeyFrames.end());
}

vector<MapPoint*> Map::GetAllMapPoints()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<MapPoint*>(mspMapPoints.begin(),mspMapPoints.end());
}

long unsigned int Map::MapPointsInMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspMapPoints.size();
}

long unsigned int Map::KeyFramesInMap()
{
    unique_lock<mutex> lock(mMutexMap);
    return mspKeyFrames.size();
}

vector<MapPoint*> Map::GetReferenceMapPoints()
{
    unique_lock<mutex> lock(mMutexMap);
    return mvpReferenceMapPoints;
}

long unsigned int Map::GetMaxKFid()
{
    unique_lock<mutex> lock(mMutexMap);
    return mnMaxKFid;
}

void Map::clear()
{
    for(set<MapPoint*>::iterator sit=mspMapPoints.begin(), send=mspMapPoints.end(); sit!=send; sit++)
        delete *sit;

    for(set<KeyFrame*>::iterator sit=mspKeyFrames.begin(), send=mspKeyFrames.end(); sit!=send; sit++)
        delete *sit;

    mspMapPoints.clear();
    mspKeyFrames.clear();
    mnMaxKFid = 0;
    mvpReferenceMapPoints.clear();
    mvpKeyFrameOrigins.clear();
}

} //namespace ORB_SLAM
