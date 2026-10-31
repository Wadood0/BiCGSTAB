#!/bin/bash
# Shared setup for the benchmark job scripts (slurm_bench_*.sh): finds the
# toolkit, builds our three versions and the reference. Source it as
# scripts/bench_env.sh from a job running in src/ (sbatch from src/); it moves
# the job into its own work directory.
#
# Uses from the environment:
#   ARCH        GPU architecture for nvcc   (default: native, i.e. the job's GPU)
#   CCCL_HOME   newer header-only CCCL, used if the toolkit's CUB lacks
#               cub::DeviceTransform        (default: ../deps/cccl if present)

set -euo pipefail

ARCH=${ARCH:-native}

# Each job builds in its own copy of src/: jobs on different nodes need
# different -arch, and a job must never rebuild binaries another job is
# running. Results go to $ROOT/results-raw.
ROOT=$(cd .. && pwd)
WORK=$ROOT/work/${SLURM_JOB_ID:-local-$$}
REF_BIN=$WORK/reference_build
mkdir -p "$WORK"
cp ./*.cu Makefile "$WORK/"
cp -r scripts "$WORK/"
cd "$WORK"
export ROOT WORK REF_BIN

# Lambda Stack nodes (jrcai06, server02) put CUDA in /usr; others in /usr/local/cuda.
if [ -x /usr/local/cuda/bin/nvcc ]; then
    CUDA_HOME=/usr/local/cuda
else
    CUDA_HOME=/usr
fi
NVCC=$CUDA_HOME/bin/nvcc
export CUDA_HOME NVCC ARCH

# cub::DeviceTransform arrived in CCCL 2.6/2.7; fall back to a newer checkout.
if ! grep -rqs "struct DeviceTransform" "$CUDA_HOME/include/cub/device/" 2>/dev/null; then
    CCCL_HOME=${CCCL_HOME:-$ROOT/deps/cccl}
    if [ ! -d "$CCCL_HOME/cub" ]; then
        echo "toolkit CUB has no DeviceTransform and no CCCL checkout at $CCCL_HOME" >&2
        exit 1
    fi
else
    CCCL_HOME=""
fi
export CCCL_HOME

echo "host:      $(hostname)"
echo "nvcc:      $NVCC ($("$NVCC" --version | tail -1))"
echo "arch:      $ARCH"
echo "cccl:      ${CCCL_HOME:-toolkit}"
echo "gcc:       $(gcc --version | head -1)"
echo "cpu:       $(lscpu | sed -n 's/^Model name: *//p')"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true

build_ours(){
    make -s CUDA_HOME="$CUDA_HOME" ARCH="$ARCH" CCCL_HOME="$CCCL_HOME" "$@"
}

build_reference(){
    NVCC="$NVCC" ARCH="$ARCH" ./scripts/build_reference.sh "$ROOT/reference_src/code" "$REF_BIN"
}
