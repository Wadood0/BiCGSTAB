# BiCGSTAB - builds both implementations.
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

TARGETS := main_cub main_thrust

all: $(TARGETS)

%: %.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

run: $(TARGETS)
	./main_cub
	./main_thrust

# ---- profiling (NVTX ranges are compiled in unconditionally) ----
NSYS    := $(CUDA_HOME)/bin/nsys
# --cuda-memory-usage adds overhead; drop it if you only want timings.
NSYSFLG := --trace=cuda,nvtx,osrt --force-overwrite=true

profile: profile-cub profile-thrust

profile-cub: main_cub
	$(NSYS) profile $(NSYSFLG) -o report_cub ./main_cub
	$(NSYS) stats --report nvtx_sum --format table report_cub.nsys-rep

profile-thrust: main_thrust
	$(NSYS) profile $(NSYSFLG) -o report_thrust ./main_thrust
	$(NSYS) stats --report nvtx_sum --format table report_thrust.nsys-rep

# side-by-side NVTX totals for the two implementations
compare: profile-cub profile-thrust
	@echo "reports written: report_cub.nsys-rep report_thrust.nsys-rep"
	@echo "open them in Nsight Systems for the timeline view"

# memory check - should report 0 errors for both
check: $(TARGETS)
	$(CUDA_HOME)/bin/compute-sanitizer --tool memcheck ./main_cub
	$(CUDA_HOME)/bin/compute-sanitizer --tool memcheck ./main_thrust

clean:
	rm -f $(TARGETS) report_*.nsys-rep report_*.sqlite

.PHONY: all run check clean profile profile-cub profile-thrust compare
