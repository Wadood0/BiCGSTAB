#!/bin/bash
#SBATCH --job-name=bicgstab-bench-multigpu
#SBATCH --output=bench-multigpu-%j.out
#SBATCH --error=bench-multigpu-%j.err
#SBATCH --partition=A100          # only node up with 4+ GPUs (server02, NVLink); 1 A100 job per user
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=06:00:00

# Multi-GPU scaling on 4 A100s: our nccl version on 1, 2 and 4 GPUs (plus cub
# and the reference's streams solver on 1 GPU for context), at the
# reference's sizes plus 128M and 256M, where there is enough work per GPU.
# Two sweeps: our fixed 50 iterations (x checked against cub, so the 2- and
# 4-GPU results are verified), then our versions run for the reference's
# iteration count.
#
#   sbatch slurm/slurm_bench_multigpu.sh

# Submit from src/: the job runs in the submit directory.
source ~/.bashrc
cd "$SLURM_SUBMIT_DIR"
source scripts/bench_env.sh

build_ours main_cub main_nccl main_cub_nosync main_nccl_nosync
build_reference

export NCCL_DEBUG=WARN
sizes=${SIZES:-1000000 2000000 4000000 8000000 16000000 32000000 64000000 128000000 256000000}
status=0
for mode in fixed50 matched; do
    match=0
    if [ $mode = matched ]; then match=1; fi
    IMPLS="cub nccl cub_nosync nccl_nosync ref_streams" NGPUS="1 2 4" REPS=${REPS:-3} MATCH_ITERS=$match \
    OUT=$ROOT/results-raw/multigpu-a100-$mode-$SLURM_JOB_ID \
        ./scripts/run_sweep.sh $sizes || status=1
done
exit $status
