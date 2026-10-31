#!/bin/bash
# Profile every BiCGSTAB implementation over a list of problem sizes with
# Nsight Systems, and collect the NVTX range timings into one CSV so the
# versions can be compared step by step as N grows.
#
# usage: scripts/profile_sweep.sh [N ...]      (default N: 1000 100000 1000000 10000000)
#
# Settings come from the environment:
#   IMPLS  implementations to profile          (default: "cub thrust nccl")
#   SEED   RNG seed shared by every run        (default: 12345)
#   NGPUS  GPUs for main_nccl                  (default: every visible GPU)
#   OUT    results directory                   (default: profile-<date>-<time>)
#   NSYS   nsys binary                         (default: nsys on PATH, else $CUDA_HOME/bin/nsys)
#
# Writes, under OUT:
#   reports/<impl>_N<n>.nsys-rep   open in the Nsight Systems GUI for the timeline
#   stats/<impl>_N<n>_nvtx_sum.csv            NVTX range totals for that run
#   stats/<impl>_N<n>_cuda_gpu_kern_sum.csv   per-kernel GPU time for that run
#   nvtx_all.csv                   every run's NVTX ranges in one table

set -uo pipefail

cd "$(dirname "$0")/.."    # src/, where the Makefile and binaries are

SIZES=("$@")
if [ ${#SIZES[@]} -eq 0 ]; then
    SIZES=(1000 100000 1000000 10000000)
fi
IMPLS=${IMPLS:-"cub thrust nccl"}
SEED=${SEED:-12345}
NGPUS=${NGPUS:-$(nvidia-smi -L | wc -l)}
OUT=${OUT:-profile-$(date +%Y%m%d-%H%M%S)}
NSYS=${NSYS:-$(command -v nsys || echo "${CUDA_HOME:-/usr/local/cuda}/bin/nsys")}

mkdir -p "$OUT/reports" "$OUT/stats" "$OUT/logs"

targets=""
for impl in $IMPLS; do targets="$targets main_$impl"; done
make -s $targets || exit 1

echo "impl,n,range,total_ns,instances,avg_ns,min_ns,max_ns" > "$OUT/nvtx_all.csv"
printf "%-7s %10s %14s %14s\n" impl N "solve (ms)" "iter avg (us)"

failed=0
for n in "${SIZES[@]}"; do
    for impl in $IMPLS; do
        name="${impl}_N${n}"
        if [ "$impl" = nccl ]; then
            cmd=(./main_nccl "$n" "$NGPUS" "$SEED")
        else
            cmd=("./main_$impl" "$n" "$SEED")
        fi

        if ! "$NSYS" profile --trace=cuda,nvtx,osrt --force-overwrite=true \
                -o "$OUT/reports/$name" "${cmd[@]}" > "$OUT/logs/$name.log" 2>&1; then
            printf "%-7s %10s  FAIL (see %s)\n" "$impl" "$n" "$OUT/logs/$name.log"
            failed=1
            continue
        fi

        # stats writes <prefix>_<report>.csv; the .sqlite it makes on the way
        # is only a cache, so drop it
        "$NSYS" stats --report nvtx_sum,cuda_gpu_kern_sum --format csv \
            --output "$OUT/stats/$name" "$OUT/reports/$name.nsys-rep" \
            >> "$OUT/logs/$name.log" 2>&1
        rm -f "$OUT/reports/$name.sqlite"

        # nvtx_sum columns: Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),
        # Min (ns),Max (ns),StdDev (ns),Style,Range - range names start with ':'
        awk -F, -v impl="$impl" -v n="$n" 'NR > 1 {
                range = $10; sub(/^:/, "", range)
                print impl "," n "," range "," $2 "," $3 "," $4 "," $6 "," $7
            }' "$OUT/stats/${name}_nvtx_sum.csv" >> "$OUT/nvtx_all.csv"

        awk -F, -v impl="$impl" -v n="$n" '
            $1 == impl && $2 == n && $3 == "solve:loop" { solve = $4 / 1e6 }
            $1 == impl && $2 == n && $3 == "iter"       { iter = $6 / 1e3 }
            END { printf "%-7s %10s %14.3f %14.1f\n", impl, n, solve, iter }' \
            "$OUT/nvtx_all.csv"
    done
done

echo
echo "results in $OUT/ (nvtx_all.csv, stats/, reports/)"
exit $failed
