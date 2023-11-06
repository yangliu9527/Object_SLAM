# Object-aware data association for the semantically constrained visual SLAM
**Authors:** Liu Yang

This is an object-level semantic visual system, which is based on [ORB_SLAM2](https://github.com/raulmur/ORB_SLAM2) and supports RGB-D and Stereo modes. We are aimed to solve two problems: the object-level data association and the semantically constrained pose optimization.  The code is an open source version implementation of our work.

```
@article{liu2023object,
  title={Object-aware data association for the semantically constrained visual SLAM},
  author={Liu, Yang and Guo, Chi and Wang, Yingli},
  journal={Intelligent Service Robotics},
  volume={16},
  number={2},
  pages={155--176},
  year={2023},
  publisher={Springer}
}
```

![image](https://github.com/yangliu9527/Object_SLAM/blob/master/example_picutures/ObjectMappingKITTI1.gif)

![image](https://github.com/yangliu9527/Object_SLAM/blob/master/example_picutures/ObjectMappingTUM_fr2_desk.gif)

# 1. Prerequisites
We have tested the library in **Ubuntu 18.04**, but it should be easy to compile in other platforms. A powerful computer (e.g. i7) will ensure real-time performance and provide more stable and accurate results.

## C++11 or C++0x Compiler
We use the new thread and chrono functionalities of C++11.

## Pangolin
We use [Pangolin](https://github.com/stevenlovegrove/Pangolin) for visualization and user interface. Dowload and install instructions can be found at: https://github.com/stevenlovegrove/Pangolin.

## OpenCV
We use [OpenCV](http://opencv.org) to manipulate images and features. Dowload and install instructions can be found at: http://opencv.org. **Required at leat 2.4.3. Tested with OpenCV 2.4.11 and OpenCV 3.2**.

## Eigen3
Required by g2o (see below). Download and install instructions can be found at: http://eigen.tuxfamily.org. **Required at least 3.1.0**.

## DBoW2 and g2o (Included in Thirdparty folder)
We use modified versions of the [DBoW2](https://github.com/dorian3d/DBoW2) library to perform place recognition and [g2o](https://github.com/RainerKuemmerle/g2o) library to perform non-linear optimizations. Both modified libraries (which are BSD) are included in the *Thirdparty* folder.

## PCL

We use PCL libraries to process point cloud. Dowload and install instructions can be found at: https://github.com/PointCloudLibrary/pcl and https://pointclouds.org/. We used command  `sudo apt-get install libpcl-dev`  in Ubuntu18.04 to install PCL.

# 2. Building ObjectSLAM

Clone the repository:
```
git clone https://github.com/yangliu9527/object_slam
```
There are some thirdparty files we can not upload due to the size. Users can copy them from [ORB_SLAM2](https://github.com/raulmur/ORB_SLAM2). Two directories need to be copy: Thirdparty and Vocabulary.


We provide a script `build.sh` to build the *Thirdparty* libraries and *ObjectSLAM*. Please make sure you have installed all required dependencies (see section 2). Execute:
```
cd object_slam
chmod +x build.sh
./build.sh
```

This will create **libORB_SLAM2.so**  at *lib* folder and the executables **rgbd_tum**, **stereo_kitti** in *Examples* folder.

# 3. Prepare Datasets
The system supports TUM RGB-D and KITTI-Odometry datasets. The interfaces are the same as ORB_SLAM2. In addition, our system needs semantic features extracted from the original images. We use [YOLACT](https://github.com/dbolya/yolact) to extract instance segmentation results offline and feed the results with image frames. We provide a demo sequence in BaiduDisk：

link：https://pan.baidu.com/s/1arvRnTaDZe1bgWENx4_Cdg 

code：lud0 

# 4. Run System

We provide two python scripts to run the system on different sequences. Please refer to run_exp_tum.py and run_exp_kitti.py for running commands.

It should be noted that the dataset path and the threshold of semantic score should be added into the .yaml file. 

