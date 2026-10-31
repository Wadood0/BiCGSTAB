# BiCGSTAB on one or more GPUs

Three CUDA implementations of the BiCGSTAB solver for a heptadiagonal
(7-diagonal) linear system Ax = b:

| Version       | File                  | What it uses                                                          |
|---------------|-----------------------|-----------------------------------------------------------------------|
| Thrust        | `src/main_thrust.cu`  | Thrust algorithms, one GPU                                            |
| CUB           | `src/main_cub.cu`     | `cub::DeviceTransform` / `TransformReduce`, two CUDA streams, one GPU |
| NCCL          | `src/main_nccl.cu`    | rows split across every GPU on the node, halo exchange and allreduce with NCCL |

All three run 50 iterations on the same random, diagonally dominant system for
a given `N` and seed, and write their solution to `x_<version>.txt`.

## Requirements

- Linux or WSL2 (native Windows nvcc isn't supported).
- CUDA Toolkit 13. The CUB version needs `cub::DeviceTransform`. The Makefile
  uses `/usr/local/cuda`, not the old nvcc that Ubuntu's `nvidia-cuda-toolkit`
  package installs. A toolkit with an older CUB can use a header-only CCCL
  checkout instead: `make CCCL_HOME=/path/to/cccl`.
- NCCL, for the NCCL version only. Leave `NCCL_HOME` unset if it's installed in
  the system paths, otherwise point it at the install prefix.

## Building

```bash
cd src
make                             # main_cub, main_thrust, main_nccl
make main_cub                    # just one
make NCCL_HOME=$HOME/.local/nccl # NCCL outside the system paths
make ARCH=sm_80                  # no GPU on this machine (e.g. a login node)
make main_cub_nosync             # build without per-step profiling syncs (any version)
make clean
```

`ARCH` defaults to `native`, which builds for the GPU in the machine.

## Running

```bash
./main_thrust [N] [SEED]
./main_cub    [N] [SEED]
./main_nccl   [N] [GPUS] [SEED]
```

| Argument | Default           | Meaning                                       |
|----------|-------------------|-----------------------------------------------|
| `N`      | 100               | number of rows (at least 7)                   |
| `GPUS`   | every visible GPU | NCCL only: GPUs to split the rows over (`N / GPUS` must be at least 3) |
| `SEED`   | current time      | random seed. Pass the same seed to every version you want to compare |

```bash
./main_cub  1000000 42
./main_nccl 1000000 4 42         # the same system, split over 4 GPUs
```

## Checking the versions agree

```bash
make verify                      # N=100, SEED=12345
make verify N=1000000 SEED=7
make verify VERIFY_IMPLS=thrust  # no NCCL on this machine
make check                       # compute-sanitizer memcheck on all three
```

`make verify` compares every entry of each version's x with `main_cub`'s. The
versions sum their dot products in different orders, so they agree to rounding
(about 1e-16), not bit for bit.

## Benchmarking over many sizes

```bash
cd src
scripts/run_sweep.sh                         # 1M, 2M, ..., 64M rows, 3 repetitions
scripts/run_sweep.sh 1000000 16000000        # your own sizes
IMPLS="cub thrust" REPS=5 scripts/run_sweep.sh
NGPUS="1 2 4" IMPLS=nccl scripts/run_sweep.sh   # NCCL scaling
```

This records the solve time per iteration of each run, checks every x against
`main_cub`'s, and writes `summary.csv` and the logs to `sweep-<date>-<time>/`
(or `OUT=<dir>`). The header of `scripts/run_sweep.sh` lists all the options,
including the no-sync builds and `MATCH_ITERS`.

### Against the reference implementation

The baseline is Dr Ayaz Khan's BiCGSTAB
([Research-Resources/bicgstab](https://github.com/Research-Resources/bicgstab)).
Clone it next to `src/` as `reference_src/`, then build it. Its sources are
copied before they are built, never modified.

```bash
git clone https://github.com/Research-Resources/bicgstab.git reference_src
cd src
scripts/build_reference.sh       # ref_streams, ref_gpu, ref_omp in ../reference_build
IMPLS="cub ref_streams ref_gpu" scripts/run_sweep.sh
```

## Profiling

Every version has NVTX ranges for each solver step.

```bash
make profile-cub                 # Nsight Systems profile + NVTX summary (profile-thrust, profile-nccl)
scripts/profile_sweep.sh         # profile every version at several N, one CSV of NVTX timings
```

By default each step waits for the GPU, so the ranges time the GPU work. The
`*_nosync` builds remove those waits, which gives the true speed at small N.

## Running on a Slurm cluster

Submit from `src/`, since the jobs run in the submit directory. First edit the
lines marked `CHANGE` (partition, account, modules) in the job scripts.

```bash
cd src
sbatch slurm/slurm_cub.sh
sbatch slurm/slurm_thrust.sh
sbatch --export=ALL,N=10000000 slurm/slurm_nccl.sh
```

The `slurm_bench_*.sh` jobs ran the benchmarks on the KFUPM JRCAI cluster:

| Job                      | What it runs                                        |
|--------------------------|-----------------------------------------------------|
| `slurm_bench_gpu.sh`     | all versions vs the reference on one RTX 3090       |
| `slurm_bench_cpu.sh`     | the reference's sequential and OpenMP CPU solvers   |
| `slurm_bench_multigpu.sh`| NCCL on 1, 2 and 4 A100s                             |

They build everything in a work directory per job and need `reference_src/`.
To merge their results into one CSV and draw the graphs:

```bash
python scripts/analyze_results.py ../results-raw results.csv figures/
```

## Layout

```
src/
  main_thrust.cu, main_cub.cu, main_nccl.cu
  Makefile           build, run, verify, check, profile
  scripts/           run_sweep, profile_sweep, build_reference, bench_env, analyze_results
  slurm/             Slurm job scripts
```
