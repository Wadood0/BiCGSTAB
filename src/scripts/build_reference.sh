#!/bin/bash
# Build Dr Ayaz Khan's reference BiCGSTAB (reference_src, github.com/Research-Resources/bicgstab)
# so it can be benchmarked next to ours. The original sources are never
# modified: they are copied into a build directory first.
#
# usage: scripts/build_reference.sh [REF_SRC] [OUT]   (relative paths are from src/)
#   REF_SRC  reference code directory   (default: ../reference_src/code)
#   OUT      where the binaries go       (default: ../reference_build)
#
# Settings from the environment:
#   NVCC   nvcc to use        (default: nvcc on PATH)
#   ARCH   GPU architecture   (default: native)
#
# Produces, in OUT:
#   ref_streams  bicgstab_clean.cu as published (CUDA_STREAMS on: 5-stream solver)
#   ref_gpu      the same file with CUDA_STREAMS off (single-stream solver), the
#                paper's "GPU without streams" configuration
#   ref_omp      bicgstab_openmp.c (sequential + OpenMP CPU solver), built as
#                the reference's compile_openmp does
#
# The two CUDA builds get two extra timer lines around the solver's iteration
# loop, which print "loop_ms=<ms>" to stderr. The reference's own timing,
# which also covers work-vector allocation, cuBLAS handle creation and the
# init kernel, is still printed on stdout unchanged. Nothing else differs.

set -euo pipefail

cd "$(dirname "$0")/.."    # src/

REF_SRC=${1:-../reference_src/code}
OUT=${2:-../reference_build}
NVCC=${NVCC:-nvcc}
ARCH=${ARCH:-native}

mkdir -p "$OUT"

# The reference includes the NVTX v2 header <nvToolsExt.h>. Recent toolkits
# ship only the header-only NVTX3 copy of it, which needs no -lnvToolsExt.
NVTX_INC=""
for d in /usr/include/nvtx3 /usr/local/cuda/include/nvtx3 \
         "$(dirname "$(command -v "$NVCC")")/../include/nvtx3"; do
    if [ -f "$d/nvToolsExt.h" ]; then NVTX_INC="-I$d"; break; fi
done

# Timer lines: start after "int k;" and stop just before the solver prints
# its iteration count. Both patterns occur once in each CUDA solver function.
# Carriage returns are stripped first: a Windows checkout has CRLF endings,
# which would stop the patterns from matching.
instrument(){
    tr -d '\r' < "$1" | sed -e 's|^    int k;$|    int k;\n    struct timeval loop_t0, loop_t1;\n    cudaCheck(cudaDeviceSynchronize());\n    gettimeofday(\&loop_t0, NULL);|' \
        -e 's|^    printf("%d,%.6f,", k, err);$|    cudaCheck(cudaDeviceSynchronize());\n    gettimeofday(\&loop_t1, NULL);\n    fprintf(stderr, "loop_ms=%.3f\\n", (loop_t1.tv_sec - loop_t0.tv_sec) * 1000.0 + (loop_t1.tv_usec - loop_t0.tv_usec) / 1000.0);\n    printf("%d,%.6f,", k, err);|'
}

instrument "$REF_SRC/bicgstab_clean.cu" > "$OUT/ref_streams.cu"
sed -e 's|^#define CUDA_STREAMS$|// #define CUDA_STREAMS   (disabled by build_reference.sh: single-stream build)|' \
    "$OUT/ref_streams.cu" > "$OUT/ref_gpu.cu"

for v in ref_streams ref_gpu; do
    if [ "$(grep -c 'loop_ms=' "$OUT/$v.cu")" -ne 2 ]; then
        echo "instrumenting $v.cu failed (expected 2 timers)" >&2
        exit 1
    fi
done
grep -q '^// #define CUDA_STREAMS' "$OUT/ref_gpu.cu" || { echo "could not disable CUDA_STREAMS" >&2; exit 1; }

# Flags as in the reference makefile (-O3 -lcublas), arch made configurable.
"$NVCC" -O3 -arch="$ARCH" $NVTX_INC "$OUT/ref_streams.cu" -o "$OUT/ref_streams" -lcublas
"$NVCC" -O3 -arch="$ARCH" $NVTX_INC "$OUT/ref_gpu.cu"     -o "$OUT/ref_gpu"     -lcublas

# As in compile_openmp: gcc -O3 -fopenmp ... -lm (NVTX calls are compiled out).
gcc -O3 -fopenmp $NVTX_INC "$REF_SRC/bicgstab_openmp.c" -o "$OUT/ref_omp" -lm

echo "reference binaries in $OUT: ref_streams ref_gpu ref_omp"
