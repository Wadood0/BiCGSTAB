#!/bin/bash
#SBATCH --job-name=bicgstab-nccl
#SBATCH --output=bicgstab-nccl-%j.out
#SBATCH --error=bicgstab-nccl-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1               # one process drives every GPU on the node
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:4             # CHANGE: GPUs per node on your cluster
#SBATCH --partition=gpu          # CHANGE: your cluster's GPU partition
##SBATCH --account=your_account  # CHANGE + uncomment if your cluster needs one

set -euo pipefail

module purge
module load cuda                 # CHANGE: e.g. cuda/13.3 - `module avail cuda` to list
module load nccl                 # CHANGE: e.g. nccl/2.27 - `module avail nccl` to list

cd "$SLURM_SUBMIT_DIR"

# Problem size; override at submit time with:  sbatch --export=ALL,N=1000000 slurm_nccl.sh
N=${N:-100}
# Use every GPU slurm gave us.
NGPUS=${SLURM_GPUS_ON_NODE:-$(nvidia-smi -L | wc -l)}

# WARN prints only real problems; set INFO to see which transport (NVLink,
# PCIe P2P, shared memory) NCCL picked between the GPUs.
export NCCL_DEBUG=WARN

echo "node:  $(hostname)"
echo "start: $(date)"
echo "gpus:  $NGPUS   N: $N"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
nvidia-smi topo -m

# Built on the allocated node so -arch=native matches the GPU we actually got.
# If your compute nodes cannot compile, build on the login node with an
# explicit arch instead and delete this line. NCCL_HOME comes from the nccl
# module; if yours does not set it, pass NCCL_HOME=... here.
make main_nccl

srun ./main_nccl "$N" "$NGPUS"

echo "end:   $(date)"
