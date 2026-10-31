# BiCGSTAB - builds all three implementations.
#
# Ubuntu's nvidia-cuda-toolkit package puts an old nvcc (CUDA 12.0) at
# /usr/bin/nvcc; it lacks cub::DeviceTransform, so main_cub will not build
# with it. Default to the real toolkit instead.
CUDA_HOME ?= /usr/local/cuda
NVCC      := $(CUDA_HOME)/bin/nvcc

# 'native' targets the GPU in this machine. On a cluster login node with no
# GPU, pass an explicit arch:  make ARCH=sm_80
ARCH      ?= native

NVCCFLAGS := -std=c++17 --extended-lambda -O3 -lineinfo -arch=$(ARCH)

# A toolkit whose bundled CUB predates cub::DeviceTransform can use a newer,
# header-only CCCL checkout instead:  make CCCL_HOME=/path/to/cccl
CCCL_HOME ?=
ifneq ($(CCCL_HOME),)
NVCCFLAGS += -I$(CCCL_HOME)/thrust -I$(CCCL_HOME)/cub -I$(CCCL_HOME)/libcudacxx/include
endif

# NCCL (main_nccl only). The libnccl2/libnccl-dev packages install to the
# default paths, so leave this empty there. Cluster nccl modules usually export
# NCCL_HOME themselves; otherwise pass it:  make NCCL_HOME=/path/to/nccl
NCCL_HOME ?=
ifneq ($(NCCL_HOME),)
NCCLFLAGS := -I$(NCCL_HOME)/include -L$(NCCL_HOME)/lib -Xlinker -rpath=$(NCCL_HOME)/lib
endif

# Per-step syncs inside the solve loop (they make the NVTX step ranges time
# GPU work). 0 drops them:  make PROFILE_SYNC=0. The *_nosync targets build
# that variant next to the normal binaries, e.g. make main_cub_nosync.
PROFILE_SYNC ?= 1
NVCCFLAGS += -DPROFILE_SYNC=$(PROFILE_SYNC)

TARGETS := main_cub main_thrust main_nccl

all: $(TARGETS)

%: %.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

main_nccl: main_nccl.cu
	$(NVCC) $(NVCCFLAGS) $(NCCLFLAGS) $< -o $@ -lnccl

%_nosync: %.cu
	$(NVCC) $(filter-out -DPROFILE_SYNC=%,$(NVCCFLAGS)) -DPROFILE_SYNC=0 $< -o $@

main_nccl_nosync: main_nccl.cu
	$(NVCC) $(filter-out -DPROFILE_SYNC=%,$(NVCCFLAGS)) $(NCCLFLAGS) -DPROFILE_SYNC=0 $< -o $@ -lnccl

run: $(TARGETS)
	./main_cub
	./main_thrust
	./main_nccl

# ---- profiling (NVTX ranges are compiled in unconditionally) ----
NSYS    := $(CUDA_HOME)/bin/nsys
# --cuda-memory-usage adds overhead; drop it if you only want timings.
NSYSFLG := --trace=cuda,nvtx,osrt --force-overwrite=true

profile: profile-cub profile-thrust profile-nccl

profile-cub: main_cub
	$(NSYS) profile $(NSYSFLG) -o report_cub ./main_cub
	$(NSYS) stats --report nvtx_sum --format table report_cub.nsys-rep

profile-thrust: main_thrust
	$(NSYS) profile $(NSYSFLG) -o report_thrust ./main_thrust
	$(NSYS) stats --report nvtx_sum --format table report_thrust.nsys-rep

# one process drives every visible GPU; NCCL's kernels show up under the cuda
# trace, and the comm:halo ranges mark the neighbour exchanges.
profile-nccl: main_nccl
	$(NSYS) profile $(NSYSFLG) -o report_nccl ./main_nccl
	$(NSYS) stats --report nvtx_sum --format table report_nccl.nsys-rep

# side-by-side NVTX totals for the three implementations
compare: profile-cub profile-thrust profile-nccl
	@echo "reports written: report_cub.nsys-rep report_thrust.nsys-rep report_nccl.nsys-rep"
	@echo "open them in Nsight Systems for the timeline view"

# memory check - should report 0 errors for all three
check: $(TARGETS)
	$(CUDA_HOME)/bin/compute-sanitizer --tool memcheck ./main_cub
	$(CUDA_HOME)/bin/compute-sanitizer --tool memcheck ./main_thrust
	$(CUDA_HOME)/bin/compute-sanitizer --tool memcheck ./main_nccl

# ---- correctness: every implementation must converge to the same x ----
# All runs get the same N and seed, so they solve the same system. Each one
# writes x_<impl>.txt, and every entry is compared against main_cub's: the
# difference must be within TOL, relative to the entry (absolute below 1).
# Reduction order differs between implementations (and GPU counts), so the
# results agree to rounding, not bit for bit.
# No NCCL on this machine?  make verify VERIFY_IMPLS=thrust
N            ?= 100
SEED         ?= 12345
NGPUS        ?= $(shell nvidia-smi -L | wc -l)
TOL          ?= 1e-8
VERIFY_IMPLS ?= thrust nccl

verify: main_cub $(addprefix main_,$(VERIFY_IMPLS))
	./main_cub $(N) $(SEED) > /dev/null
	@set -e; for impl in $(VERIFY_IMPLS); do \
	    if [ $$impl = nccl ]; then ./main_nccl $(N) $(NGPUS) $(SEED) > /dev/null; \
	    else ./main_$$impl $(N) $(SEED) > /dev/null; fi; \
	    paste x_cub.txt x_$$impl.txt | awk -v tol=$(TOL) -v impl=$$impl ' \
	        NF != 2 || $$0 ~ /nan|inf/ { bad = 1; next } \
	        { d = $$1 - $$2; if (d < 0) d = -d; \
	          m = $$1 < 0 ? -$$1 : $$1; if (m < 1) m = 1; \
	          if (d / m > worst) worst = d / m } \
	        END { ok = !bad && worst <= tol; \
	              printf "cub vs %-6s max diff %.3g  %s\n", impl, worst, \
	                     bad ? "FAIL (length mismatch or nan/inf)" : ok ? "ok" : "FAIL"; \
	              exit !ok }'; \
	done

clean:
	rm -f $(TARGETS) $(TARGETS:=_nosync) report_*.nsys-rep report_*.sqlite x_*.txt

.PHONY: all run check clean profile profile-cub profile-thrust profile-nccl compare verify
