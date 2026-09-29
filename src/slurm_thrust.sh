#!/bin/bash
#SBATCH --job-name=bicgstab-thrust
#SBATCH --output=bicgstab-thrust-%j.out
#SBATCH --error=bicgstab-thrust-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu          # CHANGE: your cluster's GPU partition
##SBATCH --account=your_account  # CHANGE + uncomment if your cluster needs one

set -euo pipefail

module purge
module load cuda                 # CHANGE: e.g. cuda/13.3 - `module avail cuda` to list

cd "$SLURM_SUBMIT_DIR"

echo "node:  $(hostname)"
echo "start: $(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# Built on the allocated node so -arch=native matches the GPU we actually got.
# If your compute nodes cannot compile, build on the login node with an
# explicit arch instead and delete this line.
make main_thrust

srun ./main_thrust

echo "end:   $(date)"
