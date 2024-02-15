#!/bin/bash
#
#SBATCH --job-name="000-92-lowlevel"
#SBATCH --partition=iris
#SBATCH --account=iris
#SBATCH --output=/iris/u/wuqi23/doggybot/output/000-92-%j.out
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1 
#SBATCH --time=12:00:00 # Max job length is 0.5 day
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
echo "task description:
    random delay [0,2,1] for action
    change observation from delta yaw to omega [-0.7,0.7]
    lin_vel_clip = 0.1
    lin_vel_x = [-0.5, 1.5]
    lin vel reward use base vel
    delay 1
    rest offset = 0"
    # add 6 zeros in priv
    # add 0.02s delay in sim 
    # detect height part*0 in obs, n_scan =132
    # no 17 stuff"
#     duplicate
#     high between finger 0.5
#     approach with stress on z:
#         err = self.target_position - self.ee_pos
#         err[:,2]*=5
# --------------------------------------------"

srun bash -c '/sailhome/wuqi23/anaconda3/envs/parkour/bin/python train.py  --task go1gp --exptid 000-92-rnddelay --device cuda:0'

# done
echo "Done"
