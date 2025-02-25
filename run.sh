python train_rlgames.py --task=[BlockAssemblySearch / BlockAssemblyOrient / BlockAssemblyGraspSim / BlockAssemblyInsertSim]   --num_envs=1
python train_rlgames.py --task=BlockAssemblySearch --num_envs=1
python train_rlgames.py --task=InspireBlockAssemblySearch --num_envs=1
python train_rlgames.py --task=InspireGraspBlock --num_envs=2

#for evaluation
python train_rlgames.py --task=BlockAssemblyGraspSim  --checkpoint=./checkpoint/block_assembly/last_AllegroHandLegoTestPAISim_ep_19000_rew_1530.9819.pth --play --num_envs=256
#in this case
python train_rlgames.py --task=InspireGraspBlock  --checkpoint=./runs/InspireGraspBlock_03-17-25-31/nn/InspireGraspBlock_best_ep_8308_rew_[60.62].pth --play --num_envs=16384


python -m debugpy --listen 34567 --wait-for-client train_rlgames.py --task=InspireBlockAssemblySearch --num_envs=1


# to run InspireGraspBlockV3 (GEN72 arm + inspire hand to grasp one lego block)
python train_rlgames.py --task=InspireGraspBlockV3 --num_envs=8
# to see InspireGraspBlockV3 evaluation
python train_rlgames.py --task=InspireGraspBlockV3 --checkpoint=./runs/InspireGraspBlockV3_19-19-40-23/nn/InspireGraspBlockV3_best_ep_45128_rew_[43.99].pth --play --num_envs=8