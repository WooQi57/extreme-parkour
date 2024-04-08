#!/bin/bash
#
#SBATCH --job-name="100-13-cam"
#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --output=/iris/u/wuqi23/doggybot/output/100-13-%j.out
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1 
#SBATCH --time=24:00:00 # Max job length is 1 day
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=32G
#SBATCH --exclude=iris-hp-z8,iris1,iris2,iris3,iris4

###SBATCH --mem-per-cpu=2G

# only use the following if you want email notification
#SBATCH --mail-user=wuqi23@cs.stanford.edu
#SBATCH --mail-type=ALL

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR
nvidia-smi

# sample process
# srun bash -c '/sailhome/wuqi23/anaconda3/envs/parkour/bin/python /iris/u/wuqi23/doggybot/test.py'
echo "
--------------------------------------------
--------------------------------------------
task description:
    use_camera
    resumed from 106-12 15000
    lin_vel_x_parkour = [0.5, 1.2] # min max [m/s]
    adjust weights
    use global vel_z 
    add terrain level reward
    resume from 102-12
    use const dist for terrain level instead of v*t
    no pitch for climbing
    vel z reward 1.25 vel max 2
    resum from 100-12
    resume from 103-10 2200
    tracking_goal_vel = 4.0 
    #resume from 101-08
    use feet height for delta_z
    _reward_lin_vel_z_parkour conditioned on close to edges or not
    lin_vel_x_parkour = [0.75, 1.5]

    #_reward_lin_vel_z_parkour becomes 1 if not climbing
    entropy coeff 0.01
    make tracking z zero for walking
    adjust parkour vx and success threshold
    use separate vx for walking and parkour
    increase collision penalty for climbing 10
    add z vel penalty for walking
    more cutoff
    add tracking_z (this is real)
    remove delta yaw (switch to omega)
    remove foot contact
    original random delay [1,1] for every 3k steps
    run parkour and cmd at the same time
--------------------------------------------
--------------------------------------------
--------------------------------------------" 

srun bash -c '/sailhome/wuqi23/anaconda3/envs/parkour/bin/python train.py  --task go2 --exptid 100-13-cam --device cuda:0 --resume --resumeid 106-12 --checkpoint 15000 --use_camera
'

# done
echo "Done"