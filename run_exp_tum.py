import os

dataset_path = "/home/zhiyu/DataSet/TUM/"
seq_paths = os.listdir(dataset_path)
#seq_paths = [path for path in seq_paths if(os.path.isdir(f'{dataset_path}/{path}'))]
#seq_paths = ["rgbd_dataset_freiburg1_desk"]
#seq_paths = ["rgbd_dataset_freiburg1_desk2"]
#seq_paths = ["rgbd_dataset_freiburg1_room"]
#seq_paths = ["rgbd_dataset_freiburg1_xyz"]
#seq_paths = ["rgbd_dataset_freiburg1_teddy"]
seq_paths = ["rgbd_dataset_freiburg2_desk"]
#seq_paths = ["rgbd_dataset_freiburg3_long_office_household"]
#seq_paths = ["rgbd_dataset_freiburg3_teddy"]

loc_eval_path = "ExpResults/TUM/Localization"
ass_eval_path = "ExpResults/TUM/ObjectDataAssociation"


for seq_path in seq_paths:
    print(f'Run the system on {seq_path}...')
    os.system(f'./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt {dataset_path}/{seq_path}/*.yaml  {dataset_path}/{seq_path} {dataset_path}/{seq_path}/associate.txt')
    os.system(f'mv CameraTrajectory.txt {loc_eval_path}/results/CameraTrajectory_{seq_path}.txt')
    os.system(f'python2 {loc_eval_path}/evaluate_ate.py {dataset_path}/{seq_path}/groundtruth.txt {loc_eval_path}/results/CameraTrajectory_{seq_path}.txt --verbose --output_path={loc_eval_path}/results/ --output_name=ATE_{seq_path}.txt --plot={loc_eval_path}/results/ATE_{seq_path}.png')
    os.system(f'python2 {loc_eval_path}/evaluate_rpe.py {dataset_path}/{seq_path}/groundtruth.txt {loc_eval_path}/results/CameraTrajectory_{seq_path}.txt --verbose --output_path={loc_eval_path}/results/ --output_name=RPE_{seq_path}.txt')
    
    os.system(f'cp ObjectInfo.txt {ass_eval_path}/ObjectInfo_{seq_path}.txt')
    os.system(f'mv ObjectInfo.txt {loc_eval_path}/ObjectInfo_{seq_path}.txt')
    print(f'Run the system on {seq_path} done.')