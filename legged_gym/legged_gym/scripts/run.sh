#!/bin/bash
#
#SBATCH --job-name="101-65-cam2ac"
#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --output=/iris/u/wuqi23/doggybot/output/101-65-%j.out
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1 
#SBATCH --time=72:00:00 # Max job length is 3 day
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=32G
#SBATCH --exclude=iris-hp-z8,iris1,iris2,iris3,iris4,iris-hgx-1

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
srun bash -c '/iris/u/wuqi23/anaconda3/envs/doggy/bin/python /iris/u/wuqi23/doggybot/test.py'
echo "
--------------------------------------------
--------------------------------------------
task description:
    # use cam
    # use 104-44 20000, 
    # num_pretrain_iter = 0
    # use 1/3 of teacher's actions for both flat and steps
    # no curriculum, step_height=0.4 + 0.25*difficulty random diff
    # terrain: use flat teacher unless near edges
    # Add yaw cutoff 0.6
    # add cur goal time out
    only forward speed
    continue_from_last_std = True
    half_valid_width=[0.8, 1.5]
    # flat only, curriculum terrain=False
    lin_vel_clip = 0.2
    reward action_rate = -0.2
    delta_torques = -2*5.0e-4
    pitch = [-0.3, 0.6]
    action_curr_step = [1,1] #[0, 1, 2, 0,] every 2000
    # randomly set vy and pitch commands to zero
    collision 20
    Pitch reward use exp(-3*err) 5
    Penalize gripper contact only
    Reward dof_error = -0.2
    Reward ang_vel_xy = -0.12 
    Friction [0.6, 5.]
    torch.abs(torch.pow(self.torques, 3))
    # reward torques = -0.00001*10
    use actual pushes
    first step farther
    narrow the step width to 0.4-0.75

    fix possibly deepcopy bug.
    fix history bug!!!
    crop [:-11, 4:-4]
    use 0.8*env_length and 0.5*env_length
    add gulf for steps
    use actor1 for depth actor init
    env class obs
    use two actors
    from scratch again
    cur_threshold_lo = 6.5
    step x range should be greater than 0.5
    step_height=0.4 + 0.15*difficulty
    found bug: obs_prop_depth[:, 5] = 0
    rest_offset = 0.01  # looks better in render if 0.005
    low y_range

    more tracking_goal_vel and less vz
    add pitch termination (abs>1.5) - useful
    use urdf v6
    cam angle 27+-5
    position = [0.3, 0, 0.147]  # front camera

    edit on_policy_runner to mask obs[5]=0 for student policy
    lin_vel_x_parkour = [0.5, 1.2] # min max [m/s] - useful
    use global vel_z 
    add terrain level reward
    use const dist for terrain level instead of v*t
    no pitch for climbing
    vel z reward 1.25 vel max 2
    tracking_goal_vel = 4.0 
    use feet height for delta_z
    _reward_lin_vel_z_parkour conditioned on close to edges or not
    lin_vel_x_parkour = [0.75, 1.5]

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
#  --resume --resumeid 104-55 --use_camera --checkpoint 20000 --resumeid_depth 104-56

srun bash -c '/iris/u/wuqi23/anaconda3/envs/doggy/bin/python train.py  --task go2 --exptid 101-65-2ac   --resume --resumeid 101-60 --checkpoint 3000 --device cuda:0
'

# done
echo "Done"