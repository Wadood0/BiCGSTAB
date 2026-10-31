#!/bin/bash
#SBATCH --job-name=bicgstab-bench-gpu
#SBATCH --output=bench-gpu-%j.out
#SBATCH --error=bench-gpu-%j.err
#SBATCH --partition=RTX3090
#SBATCH --nodelist=jrcai06        # same node for every single-GPU run (CUDA 12.8, NCCL 2.26)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

# Single-GPU comparison on one RTX 3090: our cub, thrust and nccl (1 GPU),
# each also built without per-step syncs (*_nosync),
# against the reference's GPU solver with and without CUDA streams, at the
# reference's sizes (1M..64M), REPS times each.
#
#   sbatch slurm/slurm_bench_gpu.sh                                   # ours: fixed 50 iterations
#   sbatch --export=ALL,MATCH_ITERS=1 slurm/slurm_bench_gpu.sh        # ours: the reference's count
#   sbatch --export=ALL,REPS=1,SIZES="100000 1000000" slurm/slurm_bench_gpu.sh   # quick check

# Submit from src/: the job runs in the submit directory.
source ~/.bashrc
cd "$SLURM_SUBMIT_DIR"
source scripts/bench_env.sh

build_ours main_cub main_thrust main_nccl main_cub_nosync main_thrust_nosync main_nccl_nosync
build_reference

MATCH_ITERS=${MATCH_ITERS:-0}
if [ "$MATCH_ITERS" = 1 ]; then mode=matched; else mode=fixed50; fi
IMPLS="cub thrust nccl cub_nosync thrust_nosync nccl_nosync ref_streams ref_gpu" NGPUS=1 REPS=${REPS:-5} MATCH_ITERS=$MATCH_ITERS \
OUT=$ROOT/results-raw/gpu-3090-$mode-$SLURM_JOB_ID \
    ./scripts/run_sweep.sh ${SIZES:-}
