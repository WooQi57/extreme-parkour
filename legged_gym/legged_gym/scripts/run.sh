#!/bin/bash
#
#SBATCH --job-name="000-77-lowlevel"
#SBATCH --partition=iris-hi
#SBATCH --account=iris
#SBATCH --output=/iris/u/wuqi23/doggybot/output/000-77-%j.out
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1 
#SBATCH --time=12:00:00 # Max job length is 0.5 day
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --mem=16G
#SBATCH --exclude=iris1 # Don't run on iris1

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
    duplicate
    high between finger 0.5
    approach with stress on z:
        err = self.target_position - self.ee_pos
        err[:,2]*=5
--------------------------------------------"

srun bash -c '/sailhome/wuqi23/anaconda3/envs/parkour/bin/python train.py  --task go1gp --exptid 000-77-lowlevelbl --device cuda:0'

# done
echo "Done"
