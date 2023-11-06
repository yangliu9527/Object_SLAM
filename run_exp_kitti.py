import os

dataset_path = "/home/zhiyu/DataSet/KITTI-Odometry-Full/"
eval_path = "ExpResults/KITTI/"
for i in range(0,1):
    seq_id = str(i).zfill(2)
    os.system(f'./Examples/Stereo/stereo_kitti Vocabulary/ORBvoc.txt  {dataset_path}/{seq_id}/*.yaml {dataset_path}/{seq_id}/')
    os.system(f'mv CameraTrajectory.txt {eval_path}/results/CameraTrajectory_{seq_id}.txt')
    os.system(f'mv ObjectInfo.txt {eval_path}/results/ObjectInfo_{seq_id}.txt')
    
    #if(plot_flag):
    os.system(f'evo_ape kitti {eval_path}/groundtruth/{seq_id}.txt {eval_path}/results/CameraTrajectory_{seq_id}.txt   --no_warnings  --plot_mode=xz -va --save_plot {eval_path}/results/{seq_id}_ape --save_results {eval_path}/results/{seq_id}_ape.zip')
    os.system(f'evo_rpe kitti {eval_path}/groundtruth/{seq_id}.txt {eval_path}/results/CameraTrajectory_{seq_id}.txt   --no_warnings  -r trans_part   -va --save_plot {eval_path}/results/{seq_id}_rpe_t --save_results {eval_path}/results/{seq_id}_rpe_t.zip')
    os.system(f'evo_rpe kitti {eval_path}/groundtruth/{seq_id}.txt {eval_path}/results/CameraTrajectory_{seq_id}.txt   --no_warnings  -r angle_deg   -va --save_plot {eval_path}/results/{seq_id}_rpe_R --save_results {eval_path}/results/{seq_id}_rpe_R.zip')
    os.system(f'evo_traj kitti {eval_path}/results/CameraTrajectory_{seq_id}.txt --ref={eval_path}/groundtruth/{seq_id}.txt --no_warnings   --plot_mode=xz  --save_plot {eval_path}/results/{seq_id}_traj')
    # else:
    #     os.system(f'evo_ape kitti {eval_path}/groundtruth/{seq_id}.txt {eval_path}/results/CameraTrajectory_{seq_id}.txt  --align_origin --no_warnings   -v  --save_results {eval_path}/results/{seq_id}_ape.zip')
    #     os.system(f'evo_rpe kitti {eval_path}/groundtruth/{seq_id}.txt {eval_path}/results/CameraTrajectory_{seq_id}.txt  --align_origin --no_warnings  -r trans_part   -v  --save_results {eval_path}/results/{seq_id}_rpe_t.zip')
    #     os.system(f'evo_rpe kitti {eval_path}/groundtruth/{seq_id}.txt {eval_path}/results/CameraTrajectory_{seq_id}.txt  --align_origin --no_warnings  -r angle_deg   -v  --save_results {eval_path}/results/{seq_id}_rpe_R.zip')

    os.system(f'unzip -q -o {eval_path}/results/{seq_id}_ape.zip -d {eval_path}/results/{seq_id}_ape')
    os.system(f'unzip -q -o {eval_path}/results/{seq_id}_rpe_t.zip -d {eval_path}/results/{seq_id}_rpe_t')
    os.system(f'unzip -q -o {eval_path}/results/{seq_id}_rpe_R.zip -d {eval_path}/results/{seq_id}_rpe_R')