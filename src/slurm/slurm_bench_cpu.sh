#!/bin/bash
#SBATCH --job-name=bicgstab-bench-cpu
#SBATCH --output=bench-cpu-%j.out
#SBATCH --error=bench-cpu-%j.err
#SBATCH --partition=RTX3090
#SBATCH --nodelist=jrcai06        # same host CPU as the single-GPU runs
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=47:00:00

# The reference's CPU baselines (its benchmark_openmp_NT, which repeats 10
# times): sequential and OpenMP solvers at 1M..64M, REPS times each. No GPU
# is needed.
#
#   sbatch slurm/slurm_bench_cpu.sh
#   sbatch --export=ALL,REPS=1,SIZES="100000 1000000" slurm/slurm_bench_cpu.sh   # quick check

# Submit from src/: the job runs in the submit directory.
source ~/.bashrc
cd "$SLURM_SUBMIT_DIR"
# no GPU in this job, so name the RTX 3090's architecture instead of "native"
export ARCH=sm_86
source scripts/bench_env.sh

build_reference

IMPLS="ref_omp" OMP_THREADS=${OMP_THREADS:-$SLURM_CPUS_PER_TASK} REPS=${REPS:-10} \
OUT=$ROOT/results-raw/cpu-$SLURM_JOB_ID \
    ./scripts/run_sweep.sh ${SIZES:-}
