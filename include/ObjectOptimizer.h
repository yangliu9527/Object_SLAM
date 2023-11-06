#ifndef OBJECTOPTIMIZER_H
#define OBJECTOPTIMIZER_H

#include "Map.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "LoopClosing.h"
#include "Frame.h"

#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

namespace ORB_SLAM2
{
    

extern int N_AllSemanticConstraintNum;

class ObjectOptimizer
{

public:
    int static PoseOptimization1(Frame* pFrame);//all MapPoints, fix observations
    int static PoseOptimization2(Frame* pFrame);//
    
    



    
};


}

#endif