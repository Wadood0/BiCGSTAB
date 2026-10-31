#!/bin/bash
# Benchmark the BiCGSTAB implementations over a list of problem sizes, with
# repetitions, next to Dr Ayaz Khan's reference implementation. Records the
# solve time per iteration of every run, and checks that our versions all
# converge to the same x as main_cub.
#
# usage: scripts/run_sweep.sh [N ...]      (from src/; runs there whatever the cwd)
#   default N: 1M 2M 4M 8M 16M 32M 64M, the sizes of the reference's
#   bicgstab_benchmarking.sh and benchmark_openmp_NT
#
# Settings come from the environment:
#   IMPLS         what to run                          (default: "cub thrust nccl")
#                   ours:      cub thrust nccl, and cub_nosync thrust_nosync
#                              nccl_nosync (built with PROFILE_SYNC=0: no per-step
#                              syncs in the solve loop)
#                   reference: ref_streams ref_gpu ref_omp (build with scripts/build_reference.sh)
#   REPS          repetitions of every run             (default: 3)
#   NGPUS         GPU counts to run main_nccl with     (default: every visible GPU)
#                 e.g. NGPUS="1 2 4" runs each size on 1, 2 and 4 GPUs
#   OMP_THREADS   threads for ref_omp                  (default: 16)
#   REF_BIN       reference binaries                   (default: ../reference_build)
#   SEED          RNG seed for our versions            (default: 12345)
#   TOL           allowed difference from cub's x      (default: 1e-8)
#   VERIFY_MAX_N  largest N whose x is checked         (default: 16000000; the x files
#                 are 20 bytes per row, so checking 64M rows means 1.3 GB per file)
#   MATCH_ITERS   1 = run our versions for exactly as many iterations as the
#                 reference needed in the same repetition (default: 0 = our
#                 fixed 50). The reference runs first. A timing experiment
#                 only: our system converges in ~12 iterations, and far past
#                 that rho underflows and x turns into NaN, so x is not checked.
#   OUT           results directory                    (default: sweep-<date>-<time>)
#
# Writes OUT/summary.csv (one row per run, columns below) and every run's
# output in OUT/logs. x is only checked on the first repetition.
#
# summary.csv columns:
#   impl         cub, thrust, nccl, ref_streams, ref_gpu, ref_seq, ref_omp
#   n, gpus, threads, rep
#   iterations   iterations executed (ours: always 50; reference: until its
#                stopping test |‖x‖ - ‖x_prev‖| < 1e-6, at most 1000)
#   solve_ms     time of the iteration loop alone (ours, ref_streams, ref_gpu)
#   ms_per_iter  solve_ms / iterations (reference CPU: total_ms / iterations)
#   total_ms     the reference's own timed region (allocation + cuBLAS setup +
#                loop); "-" for ours
#   wall_s       wall time of the whole process
#   result       ours: final r.r; reference: final ‖x‖ (its "err")
#   max_diff_vs_cub, status

set -uo pipefail

cd "$(dirname "$0")/.."    # src/, where the Makefile and binaries are

SIZES=("$@")
if [ ${#SIZES[@]} -eq 0 ]; then
    SIZES=(1000000 2000000 4000000 8000000 16000000 32000000 64000000)
fi
IMPLS=${IMPLS:-"cub thrust nccl"}
REPS=${REPS:-3}
NGPUS=${NGPUS:-$(nvidia-smi -L | wc -l)}
OMP_THREADS=${OMP_THREADS:-16}
REF_BIN=${REF_BIN:-../reference_build}
SEED=${SEED:-12345}
TOL=${TOL:-1e-8}
VERIFY_MAX_N=${VERIFY_MAX_N:-16000000}
OUT=${OUT:-sweep-$(date +%Y%m%d-%H%M%S)}
MATCH_ITERS=${MATCH_ITERS:-0}

# Every x is compared against cub's, so cub runs first when it is selected.
if [[ " $IMPLS " == *" cub "* ]]; then
    IMPLS="cub $(echo " $IMPLS " | sed 's/ cub / /g')"
fi
# Matching iteration counts needs the reference's count first.
if [ "$MATCH_ITERS" = 1 ]; then
    refs="" rest=""
    for impl in $IMPLS; do
        case $impl in ref_streams|ref_gpu) refs="$refs $impl" ;; *) rest="$rest $impl" ;; esac
    done
    if [ -z "$refs" ]; then
        echo "MATCH_ITERS=1 needs ref_streams or ref_gpu in IMPLS" >&2
        exit 1
    fi
    IMPLS="$refs $rest"
fi

mkdir -p "$OUT/logs"

# make only rebuilds what is out of date; the reference is built separately
targets=""
for impl in $IMPLS; do
    case $impl in
        cub|thrust|nccl|cub_nosync|thrust_nosync|nccl_nosync) targets="$targets main_$impl" ;;
        ref_streams|ref_gpu|ref_omp)
            if [ ! -x "$REF_BIN/$impl" ]; then
                echo "$REF_BIN/$impl not found - run scripts/build_reference.sh first" >&2
                exit 1
            fi ;;
        *) echo "unknown implementation: $impl" >&2; exit 1 ;;
    esac
done
if [ -n "$targets" ]; then
    make -s $targets || exit 1
fi

# Largest relative difference between two x files (absolute for entries below
# 1), or "mismatch" when the lengths differ or either holds nan/inf.
max_diff(){
    paste "$1" "$2" | awk '
        NF != 2 || $0 ~ /nan|inf/ { bad = 1; next }
        { d = $1 - $2; if (d < 0) d = -d
          m = $1 < 0 ? -$1 : $1; if (m < 1) m = 1
          if (d / m > worst) worst = d / m }
        END { if (bad) print "mismatch"; else printf "%.3g\n", worst }'
}

# The reference prints the loop index k it stopped at: k + 1 iterations ran
# if its stopping test fired, MAX_ITER (1000) if it never did.
ref_iters(){
    awk -v k="$1" 'BEGIN { k += 0; print (k < 1000 ? k + 1 : 1000) }'
}

per_iter(){
    awk -v t="$1" -v i="$2" 'BEGIN { if (t == "-" || i + 0 == 0) print "-"; else printf "%.4f", t / i }'
}

csv="$OUT/summary.csv"
echo "impl,n,gpus,threads,rep,iterations,solve_ms,ms_per_iter,total_ms,wall_s,result,max_diff_vs_cub,status" > "$csv"
printf "%-11s %9s %4s %4s %3s %5s %11s %10s %11s  %s\n" \
    impl N gpus thr rep iters "solve ms" "ms/iter" "total ms" status

row(){   # impl n gpus threads rep iters solve_ms total_ms wall result diff status
    local mpi
    if [ "$7" != "-" ]; then mpi=$(per_iter "$7" "$6"); else mpi=$(per_iter "$8" "$6"); fi
    printf "%-11s %9s %4s %4s %3s %5s %11s %10s %11s  %s\n" "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$mpi" "$8" "${12}"
    echo "$1,$2,$3,$4,$5,$6,$7,$mpi,$8,$9,${10},${11},${12}" >> "$csv"
}

for n in "${SIZES[@]}"; do
    for rep in $(seq 1 "$REPS"); do
        verify=0
        if [ "$rep" -eq 1 ] && [ "$n" -le "$VERIFY_MAX_N" ] && [ "$MATCH_ITERS" != 1 ]; then verify=1; fi
        cub_x="$OUT/x_cub.tmp"
        rm -f "$cub_x"
        match_k=""   # the reference's iteration count in this repetition

        for impl in $IMPLS; do
            case $impl in
                nccl|nccl_nosync) gpu_counts=$NGPUS ;;
                *)    gpu_counts=1 ;;
            esac
            for gpus in $gpu_counts; do
                tag="${impl}_N${n}_g${gpus}_r${rep}"
                log="$OUT/logs/$tag.log"
                case $impl in
                    cub|thrust|cub_nosync|thrust_nosync) cmd=("./main_$impl" "$n" "$SEED") ;;
                    nccl|nccl_nosync) cmd=("./main_$impl" "$n" "$gpus" "$SEED") ;;
                    ref_streams|ref_gpu) cmd=("$REF_BIN/$impl" "$n") ;;
                    ref_omp)    cmd=("$REF_BIN/ref_omp" "$n" "$OMP_THREADS") ;;
                esac
                case $impl in
                    cub|thrust|nccl|cub_nosync|thrust_nosync|nccl_nosync)
                        if [ "$MATCH_ITERS" = 1 ]; then
                            if [ -z "$match_k" ]; then
                                row "$impl" "$n" "$gpus" - "$rep" - - - - - - "FAIL (no reference iteration count)"
                                continue
                            fi
                            cmd+=("$match_k")
                        fi ;;
                esac

                xf="x_${impl%_nosync}.txt"   # a *_nosync build writes its base name's file
                rm -f "$xf"
                start=$(date +%s.%N)
                "${cmd[@]}" > "$log" 2>&1
                rc=$?
                end=$(date +%s.%N)
                wall=$(awk -v a="$start" -v b="$end" 'BEGIN { printf "%.2f", b - a }')

                case $impl in
                    cub|thrust|nccl|cub_nosync|thrust_nosync|nccl_nosync)
                        iters=$(awk '/^iterations:/ { print $2 }' "$log" | tail -1)
                        solve=$(awk '/^solve_ms:/ { print $2 }' "$log" | tail -1)
                        result=$(awk '/^residual:/ { print $2 }' "$log" | tail -1)
                        diff="-"
                        if [ $rc -ne 0 ] || [ -z "$solve" ]; then
                            status="FAIL (exit $rc; see $log)"
                        else
                            status=ok
                            if [ $verify -eq 1 ] && [ -f "$xf" ]; then
                                if [ "$impl" = cub ]; then
                                    mv "$xf" "$cub_x"
                                elif [ -f "$cub_x" ]; then
                                    diff=$(max_diff "$cub_x" "$xf")
                                    if [ "$diff" = mismatch ] || awk -v d="$diff" -v t="$TOL" 'BEGIN { exit !(d > t) }'; then
                                        status="FAIL (x differs from cub)"
                                    fi
                                fi
                            fi
                        fi
                        rm -f "$xf"
                        row "$impl" "$n" "$gpus" - "$rep" "${iters:--}" "${solve:--}" - "$wall" "${result:--}" "$diff" "$status"
                        ;;
                    ref_streams|ref_gpu)
                        # stdout: k,err,total_ms (as built, NO_SEQUENTIAL drops the n
                        # column)   stderr: loop_ms=<ms>. On a CUDA error the
                        # reference prints a message and exits 0, so the
                        # line's absence is the failure signal.
                        line=$(grep -E '^[0-9]+,[0-9.]+,[0-9.]+$' "$log" | tail -1)
                        loop=$(sed -n 's/^loop_ms=//p' "$log" | tail -1)
                        if [ $rc -ne 0 ] || [ -z "$line" ]; then
                            row "$impl" "$n" 1 - "$rep" - - - "$wall" - - "FAIL (exit $rc; see $log)"
                        else
                            IFS=, read -r k err total <<< "$line"
                            row "$impl" "$n" 1 - "$rep" "$(ref_iters "$k")" "${loop:--}" "$total" "$wall" "$err" - ok
                            match_k=${match_k:-$(ref_iters "$k")}
                        fi
                        ;;
                    ref_omp)
                        # n,NT,seqIter, seqErr,seqTime,parIter, parErr,parTime,Correct!
                        line=$(grep -E '^[0-9]+,[0-9]+,' "$log" | tail -1 | tr -d ' ')
                        if [ $rc -ne 0 ] || [ -z "$line" ]; then
                            row ref_omp "$n" 0 "$OMP_THREADS" "$rep" - - - "$wall" - - "FAIL (exit $rc; see $log)"
                        else
                            IFS=, read -r _ nt sk serr stime pk perr ptime check <<< "$line"
                            check=${check//,/;}
                            row ref_seq "$n" 0 1 "$rep" "$(ref_iters "$sk")" - "$stime" "$wall" "$serr" - ok
                            row ref_omp "$n" 0 "$nt" "$rep" "$(ref_iters "$pk")" - "$ptime" "$wall" "$perr" - "ok (${check:-?})"
                        fi
                        ;;
                esac
            done
        done
        rm -f "$cub_x"
    done
done

echo
echo "results in $OUT/ (summary.csv, logs/)"
# non-zero exit if anything failed, so a slurm job or CI step shows it
! grep -q FAIL "$csv"
