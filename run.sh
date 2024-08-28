python train_rlgames.py --task=[BlockAssemblySearch / BlockAssemblyOrient / BlockAssemblyGraspSim / BlockAssemblyInsertSim]   --num_envs=1
python train_rlgames.py --task=BlockAssemblySearch --num_envs=1
python train_rlgames.py --task=InspireBlockAssemblySearch --num_envs=1


python -m debugpy --listen 34567 --wait-for-client train_rlgames.py --task=InspireBlockAssemblySearch --num_envs=1