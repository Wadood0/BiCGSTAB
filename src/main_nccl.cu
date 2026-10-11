#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/memory.h>
#include <thrust/generate.h>
#include <thrust/random.h>
#include <stdlib.h>
#include <stdio.h>
#include <nvtx3/nvToolsExt.h>
#include <iostream>
#include <vector>
#include <initializer_list>
#include <cub/cub.cuh>
#include <cuda/std/functional>
#include <nccl.h>

// ---- CUDA error checking ----
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d in '%s': %s\n",               \
                    __FILE__, __LINE__, #call, cudaGetErrorString(err_));      \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ---- NCCL error checking ----
#define NCCL_CHECK(call)                                                       \
    do {                                                                       \
        ncclResult_t res_ = (call);                                            \
        if (res_ != ncclSuccess) {                                             \
            fprintf(stderr, "NCCL error at %s:%d in '%s': %s\n",               \
                    __FILE__, __LINE__, #call, ncclGetErrorString(res_));      \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// One process drives every GPU, so "wait for the GPU" means every GPU: make
// each device current in turn and drain its stream.
#define SYNC_ALL(gpus)                                                         \
    do {                                                                       \
        for (auto& g_ : (gpus)) {                                              \
            CUDA_CHECK(cudaSetDevice(g_.dev));                                 \
            CUDA_CHECK(cudaStreamSynchronize(g_.stream));                      \
        }                                                                      \
    } while (0)

// Kernel launches return void: check the launch here. The execution is checked
// when SYNC_ALL drains the streams - syncing after every launch, as the single
// GPU version does, would serialise the GPUs against each other.
#define CUDA_CHECK_KERNEL() CUDA_CHECK(cudaGetLastError())


// ---- NVTX profiling ranges ----
// Device work is asynchronous, so a range closed straight after a launch would
// time the launch, not the work. PROF_POP_SYNC waits for every GPU first;
// PROF_POP is for host-only regions where there is nothing to wait for.
#define PROF_PUSH(name) nvtxRangePushA(name)
#define PROF_POP()      nvtxRangePop()
#define PROF_POP_SYNC(gpus)                                                    \
    do {                                                                       \
        SYNC_ALL(gpus);                                                        \
        nvtxRangePop();                                                        \
    } while (0)


/*
ALL VECTORS MUST BE PADDED WITH THREE 0s AT THE START AND END FOR LOGICAL ACCURACY

SINCE THE MATRIX IS HEPTADIAGONAL AND IS FOLLOWING A CUSTOM FORMAT THE FIRST AND LAST THREE ROWS
MUST ALSO BE PADDED WITH 0s AS SHOWN:
000XXXX
00XXXXX
0XXXXXX
...
XXXXXX0
XXXXX00
XXXX000
*/

/*
MULTI GPU LAYOUT

THE ROWS ARE SPLIT INTO CONTIGUOUS BLOCKS, ONE PER GPU. ROW a OF A ONLY TOUCHES COLUMNS a-3 .. a+3,
SO A GPU NEEDS THE THREE ROWS EITHER SIDE OF ITS BLOCK FROM ITS NEIGHBOURS (THE HALO) BEFORE EVERY
A*p AND A*s. VECTORS THAT FEED A MATRIX-VECTOR PRODUCT (x, p, s) ARE STORED AS:

[ 3 halo from left GPU | n_local own rows | 3 halo from right GPU ]

THE OUTER HALOS OF THE FIRST AND LAST GPU ARE NEVER WRITTEN AND STAY 0, WHICH IS THE PADDING ABOVE.
DOT PRODUCTS ARE REDUCED PER GPU WITH CUB AND SUMMED ACROSS GPUs WITH AN NCCL ALLREDUCE, SO EVERY
GPU HOLDS ITS OWN COPY OF rho, alpha, omega, beta AND COMPUTES THEM ITSELF.
*/


#define ERROR_RATE 1e-8

// Halo width: the matrix reaches 3 columns either side of the diagonal.
#define HALO 3

struct multiply
{
    const double* a;
    const double* b;

    __host__ __device__
    double operator()(int i) const
    {
        return a[i] * b[i];
    }
};

// ptr_x is the halo buffer, so ptr_x[a + i] is global column row0 + a + i - 3.
// The bounds check uses the global index; the halo itself is always in range.
struct r_calc {
    const double* ptr_A;
    const double* ptr_b;
    const double* ptr_x;
    const int row0;
    const int n;

    __host__ __device__
    double operator()(int a) const {
        double temp = ptr_b[a];
        for (int i = 0; i < 7; i++){
            int j = row0 + a + i - 3;
            if (j < 0 || j >= n) continue;
            temp = temp - ptr_A[a * 7 + i] * ptr_x[a + i];
        }
        return temp;
    }
};


struct p_calc {
    const double* ptr_r;
    const double* ptr_p;
    const double* ptr_v;
    const double* beta;
    const double* omega;

    __host__ __device__
    double operator()(int a) const {
        return ptr_r[a] + (*beta) * (ptr_p[a] - (*omega) * ptr_v[a]);
    }
};

// ptr_p is the halo buffer (see r_calc).
struct v_calc {
    const double* ptr_A;
    const double* ptr_p;
    const int row0;
    const int n;

    __host__ __device__
    double operator()(int a) const {
        double temp = 0;
        for (int i = 0; i < 7; i++){
            int j = row0 + a + i - 3;
            if (j < 0 || j >= n) continue;
            temp += ptr_A[a * 7 + i] * ptr_p[a + i];
        }
        return temp;
    }
};

struct s_calc {
    const double* ptr_r;
    const double* ptr_v;
    const double* alpha;

    __host__ __device__
    double operator()(int a) const {
        return ptr_r[a] - (*alpha) * ptr_v[a];
    }
};

// ptr_s is the halo buffer (see r_calc).
struct t_calc {
    const double* ptr_A;
    const double* ptr_s;
    const int row0;
    const int n;

    __host__ __device__
    double operator()(int a) const {
        double temp = 0;
        for (int i = 0; i < 7; i++){
            int j = row0 + a + i - 3;
            if (j < 0 || j >= n) continue;
            temp += ptr_A[a * 7 + i] * ptr_s[a + i];
        }
        return temp;
    }

};

struct x_calc {
    const double* ptr_x;
    const double* ptr_p;
    const double* ptr_s;
    const double* alpha;
    const double* omega;

    __host__ __device__
    double operator()(int a) const {
        return ptr_x[a] + (*alpha) * ptr_p[a] + (*omega) * ptr_s[a];
    }
};

struct r_calc_update {
    const double* ptr_s;
    const double* ptr_t;
    const double* omega;

    __host__ __device__
    double operator()(int a) const {
        return ptr_s[a] - (*omega) * ptr_t[a];
    }

};

__global__
void set_value(double* x, double* y){
    *x = *y;
}

__global__
void div(double* x, double* y, double* z){
    *x = *y / *z;
}

__global__
void mult(double* x, double* y, double* z){
    *x = *y * *z;
}

// Everything one GPU owns: rows [row0, row0 + n_local) of the system, its
// stream and its NCCL communicator. x, p and s are halo buffers of
// n_local + 2*HALO entries; everything else is n_local long.
struct Shard {
    int dev;
    int row0;
    int n_local;
    cudaStream_t stream;
    ncclComm_t comm;

    // matrix/vectors
    double* A;
    double* b;
    double* x;
    double* r;
    double* r_hat;
    double* v;
    double* p;
    double* s;
    double* t;

    // scalars
    double* d_rho;
    double* d_alpha;
    double* d_omega;
    double* d_beta;
    double* d_rho_p;

    // Intermediary Scalars
    double* rho_alpha;
    double* rho_p_omega;
    double* r_hat_v;
    double* sTt;
    double* tTt;
    double* rTr;

    // CUB scratch
    void* d_temp_storage;
    size_t temp_bytes;
};

// Sum per-GPU partial dot products in place, so every GPU ends up with the
// global value. Several scalars in one call share a single NCCL group launch.
static void allreduce_sum(std::vector<Shard>& gpus,
                          std::initializer_list<double* Shard::*> fields){
    NCCL_CHECK(ncclGroupStart());
    for (auto field : fields){
        for (auto& g : gpus){
            NCCL_CHECK(ncclAllReduce(g.*field, g.*field, 1, ncclDouble,
                                     ncclSum, g.comm, g.stream));
        }
    }
    NCCL_CHECK(ncclGroupEnd());
}

// Fill the halos of a halo buffer from the neighbouring GPUs. Rank k sends its
// first HALO own rows left and its last HALO own rows right, and receives the
// matching rows into its own halos. All sends/recvs go in one group, otherwise
// the paired calls would deadlock waiting on each other.
static void halo_exchange(std::vector<Shard>& gpus, double* Shard::* field){
    int ngpus = gpus.size();
    NCCL_CHECK(ncclGroupStart());
    for (int k = 0; k < ngpus; k++){
        Shard& g = gpus[k];
        double* buf = g.*field;

        // own rows live at buf[HALO .. HALO + n_local)
        if (k > 0){
            NCCL_CHECK(ncclSend(buf + HALO, HALO, ncclDouble, k - 1, g.comm, g.stream));
            NCCL_CHECK(ncclRecv(buf, HALO, ncclDouble, k - 1, g.comm, g.stream));
        }
        if (k < ngpus - 1){
            NCCL_CHECK(ncclSend(buf + g.n_local, HALO, ncclDouble, k + 1, g.comm, g.stream));
            NCCL_CHECK(ncclRecv(buf + HALO + g.n_local, HALO, ncclDouble, k + 1, g.comm, g.stream));
        }
    }
    NCCL_CHECK(ncclGroupEnd());
}

// r.r across every GPU, read back from GPU 0 (all GPUs hold the same value).
static double residual_norm(std::vector<Shard>& gpus){
    thrust::counting_iterator<int> first(0);

    for (auto& g : gpus){
        CUDA_CHECK(cudaSetDevice(g.dev));
        CUDA_CHECK(cub::DeviceReduce::TransformReduce(
            g.d_temp_storage,
            g.temp_bytes,
            first,
            g.rTr,
            g.n_local,
            cuda::std::plus<double>{},
            multiply{g.r, g.r},
            0.0,
            g.stream
        ));
    }
    allreduce_sum(gpus, {&Shard::rTr});

    double h_rTr;
    CUDA_CHECK(cudaSetDevice(gpus[0].dev));
    CUDA_CHECK(cudaMemcpyAsync(&h_rTr, gpus[0].rTr, sizeof(double),
                               cudaMemcpyDeviceToHost, gpus[0].stream));
    CUDA_CHECK(cudaStreamSynchronize(gpus[0].stream));
    return h_rTr;
}

int main(int argc, char** argv){

    // Problem size: optional first argument, defaults to 100.
    int n = 100;
    if (argc > 1){
        n = atoi(argv[1]);
    }
    if (n < 7){
        std::cerr << "N must be >= 7 so the first and last three rows of the "
                     "padding scheme stay distinct; got " << n << std::endl;
        return 1;
    }

    // GPU count: optional second argument, defaults to every visible GPU.
    int dev_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    int ngpus = dev_count;
    if (argc > 2){
        ngpus = atoi(argv[2]);
    }
    if (ngpus < 1 || ngpus > dev_count){
        std::cerr << "GPU count must be between 1 and " << dev_count
                  << " (visible devices); got " << ngpus << std::endl;
        return 1;
    }
    // A halo only reaches the immediate neighbour, so every block needs at
    // least HALO rows of its own.
    if (n / ngpus < HALO){
        std::cerr << "N must be >= " << HALO * ngpus << " so each of the "
                  << ngpus << " GPUs owns at least " << HALO << " rows; got "
                  << n << std::endl;
        return 1;
    }

    PROF_PUSH("setup:host_init");
    // Initial values on host to simulate real world environment
    thrust::default_random_engine rng(time(NULL));
    thrust::uniform_real_distribution<double> dist(-1.0, 1.0);

    thrust::host_vector<double> h_A(n*7);
    thrust::host_vector<double> h_b(n);
    thrust::host_vector<double> h_x(n);

    thrust::generate(h_A.begin(), h_A.end(), [&] { return dist(rng); });
    thrust::generate(h_b.begin(), h_b.end(), [&] { return dist(rng); });
    thrust::generate(h_x.begin(), h_x.end(), [&] { return dist(rng); });

    // Zero Padding - IMPORTANT
    int zero_indices_mat[] = {0, 1, 2, 7, 8, 14, n*7-15, n*7-9, n*7-8, n*7-3, n*7-2, n*7-1};
    for (int i = 0; i < 12; i++){
        h_A[zero_indices_mat[i]] = 0.0;
    }

    // Diagonal dominance: offset 3 is the diagonal, and the six off-diagonals
    // are each < 1 in magnitude, so 8.0 dominates their sum.
    for (int a = 0; a < n; a++){
        h_A[a * 7 + 3] = 8.0;
    }

    // Row partition: the first n % ngpus GPUs take one extra row.
    std::vector<Shard> gpus(ngpus);
    int row0 = 0;
    for (int k = 0; k < ngpus; k++){
        gpus[k].dev = k;
        gpus[k].row0 = row0;
        gpus[k].n_local = n / ngpus + (k < n % ngpus ? 1 : 0);
        row0 += gpus[k].n_local;
    }

    PROF_POP(); // setup:host_init

    // One stream per GPU; CUB, the scalar kernels and NCCL all run on it, so
    // work on a GPU is ordered without any host synchronisation.
    PROF_PUSH("setup:nccl_init");
    std::vector<int> devs(ngpus);
    std::vector<ncclComm_t> comms(ngpus);
    for (int k = 0; k < ngpus; k++){
        devs[k] = gpus[k].dev;
        CUDA_CHECK(cudaSetDevice(gpus[k].dev));
        CUDA_CHECK(cudaStreamCreate(&gpus[k].stream));
    }
    // rank k <-> devs[k], which halo_exchange relies on
    NCCL_CHECK(ncclCommInitAll(comms.data(), ngpus, devs.data()));
    for (int k = 0; k < ngpus; k++){
        gpus[k].comm = comms[k];
    }
    PROF_POP(); // setup:nccl_init

    // Initial values - GPU
    // matrix/vectors
    // Raw cudaMalloc rather than device_vector: each buffer belongs to a
    // specific device and must be freed with that device current.
    PROF_PUSH("setup:device_alloc");
    for (auto& g : gpus){
        CUDA_CHECK(cudaSetDevice(g.dev));
        size_t bytes      = g.n_local * sizeof(double);
        size_t halo_bytes = (g.n_local + 2 * HALO) * sizeof(double);

        CUDA_CHECK(cudaMalloc(&g.A, 7 * bytes));
        CUDA_CHECK(cudaMalloc(&g.b, bytes));
        CUDA_CHECK(cudaMalloc(&g.x, halo_bytes));

        CUDA_CHECK(cudaMalloc(&g.r, bytes));
        CUDA_CHECK(cudaMalloc(&g.r_hat, bytes));

        CUDA_CHECK(cudaMalloc(&g.v, bytes));
        CUDA_CHECK(cudaMalloc(&g.p, halo_bytes));

        CUDA_CHECK(cudaMalloc(&g.s, halo_bytes));
        CUDA_CHECK(cudaMalloc(&g.t, bytes));

        // device_vector zero-fills; p and v must start at 0 for the first
        // p update, and the outer halos must stay 0 for the padding.
        CUDA_CHECK(cudaMemset(g.x, 0, halo_bytes));
        CUDA_CHECK(cudaMemset(g.r, 0, bytes));
        CUDA_CHECK(cudaMemset(g.r_hat, 0, bytes));
        CUDA_CHECK(cudaMemset(g.v, 0, bytes));
        CUDA_CHECK(cudaMemset(g.p, 0, halo_bytes));
        CUDA_CHECK(cudaMemset(g.s, 0, halo_bytes));
        CUDA_CHECK(cudaMemset(g.t, 0, bytes));
    }
    PROF_POP_SYNC(gpus); // setup:device_alloc

    // Each GPU gets its own slice of A, b and x0; x0 goes past the left halo.
    PROF_PUSH("setup:h2d_transfer");
    for (auto& g : gpus){
        CUDA_CHECK(cudaSetDevice(g.dev));
        CUDA_CHECK(cudaMemcpy(g.A, thrust::raw_pointer_cast(h_A.data()) + g.row0 * 7,
                              g.n_local * 7 * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(g.b, thrust::raw_pointer_cast(h_b.data()) + g.row0,
                              g.n_local * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(g.x + HALO, thrust::raw_pointer_cast(h_x.data()) + g.row0,
                              g.n_local * sizeof(double), cudaMemcpyHostToDevice));
    }
    PROF_POP_SYNC(gpus); // setup:h2d_transfer


    // scalars
    // double rho = 1.0, omega = 1.0, alpha = 1.0;
    // double beta = 0.0, rho_p = 1.0;
    double one = 1.0;
    double zero = 0.0;

    PROF_PUSH("setup:scalar_alloc");
    for (auto& g : gpus){
        CUDA_CHECK(cudaSetDevice(g.dev));
        CUDA_CHECK(cudaMalloc(&g.d_rho,   sizeof(double)));
        CUDA_CHECK(cudaMalloc(&g.d_rho_p, sizeof(double)));
        CUDA_CHECK(cudaMalloc(&g.d_alpha, sizeof(double)));
        CUDA_CHECK(cudaMalloc(&g.d_omega, sizeof(double)));
        CUDA_CHECK(cudaMalloc(&g.d_beta,  sizeof(double)));

        // Intermediary Scalars
        CUDA_CHECK(cudaMalloc(&g.rho_alpha, sizeof(double)));
        CUDA_CHECK(cudaMalloc(&g.rho_p_omega, sizeof(double)));
        CUDA_CHECK(cudaMalloc(&g.r_hat_v, sizeof(double)));
        CUDA_CHECK(cudaMalloc(&g.sTt, sizeof(double)));
        CUDA_CHECK(cudaMalloc(&g.tTt, sizeof(double)));
        CUDA_CHECK(cudaMalloc(&g.rTr, sizeof(double)));
    }
    PROF_POP_SYNC(gpus); // setup:scalar_alloc

    PROF_PUSH("setup:scalar_upload");
    for (auto& g : gpus){
        CUDA_CHECK(cudaSetDevice(g.dev));
        CUDA_CHECK(cudaMemcpy(g.d_rho,   &one, sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(g.d_rho_p, &one, sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(g.d_alpha, &one, sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(g.d_omega, &one, sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(g.d_beta,  &zero, sizeof(double), cudaMemcpyHostToDevice));
    }
    PROF_POP_SYNC(gpus); // setup:scalar_upload

    // index sequence the functors are driven by (local row index on each GPU)
    thrust::counting_iterator<int> first(0);


    // Initial calculations
    PROF_PUSH("init:r0 = b - A*x0");
    PROF_PUSH("comm:halo x");
    halo_exchange(gpus, &Shard::x);
    PROF_POP_SYNC(gpus); // comm:halo x
    for (auto& g : gpus){
        CUDA_CHECK(cudaSetDevice(g.dev));
        CUDA_CHECK(cub::DeviceTransform::Transform(
            first,
            g.r,
            g.n_local,
            r_calc{g.A, g.b, g.x, g.row0, n},
            g.stream
        ));
    }
    PROF_POP_SYNC(gpus); // init:r0 = b - A*x0

    PROF_PUSH("init:r_hat = r0");
    for (auto& g : gpus){
        CUDA_CHECK(cudaSetDevice(g.dev));
        // r_hat = r so r.r != 0
        CUDA_CHECK(cudaMemcpyAsync(g.r_hat, g.r, g.n_local * sizeof(double),
                                   cudaMemcpyDeviceToDevice, g.stream));
    }
    PROF_POP_SYNC(gpus); // init:r_hat = r0

    // Every reduction below has the same input type, op and (per GPU) length,
    // so one scratch buffer sized once per GPU serves all of them.
    PROF_PUSH("setup:cub_temp_alloc");
    for (auto& g : gpus){
        CUDA_CHECK(cudaSetDevice(g.dev));
        g.d_temp_storage = nullptr;
        g.temp_bytes = 0;

        CUDA_CHECK(cub::DeviceReduce::TransformReduce(
            g.d_temp_storage,
            g.temp_bytes,
            first,
            g.d_rho_p,
            g.n_local,
            cuda::std::plus<double>{},
            multiply{g.r_hat, g.r},
            0.0,
            g.stream
        ));

        CUDA_CHECK(cudaMalloc(&g.d_temp_storage, g.temp_bytes));
    }
    PROF_POP_SYNC(gpus); // setup:cub_temp_alloc

    // loop
    PROF_PUSH("init:residual_norm");
    std::cout << residual_norm(gpus) << std::endl;
    PROF_POP_SYNC(gpus); // init:residual_norm
    PROF_PUSH("solve:loop");
    for (int i = 0; i < 50; i++){
        PROF_PUSH("iter");
        // rho_p = rho;
        // rho = r.r_hat summed over every GPU
        // beta = (rho / rho_p) * (alpha / omega);

        PROF_PUSH("step:rho");
        for (auto& g : gpus){
            CUDA_CHECK(cudaSetDevice(g.dev));
            set_value<<<1, 1, 0, g.stream>>>(g.d_rho_p, g.d_rho);
            CUDA_CHECK_KERNEL();

            // rho (this GPU's rows only)
            CUDA_CHECK(cub::DeviceReduce::TransformReduce(
                g.d_temp_storage,
                g.temp_bytes,
                first,
                g.d_rho,
                g.n_local,
                cuda::std::plus<double>{},
                multiply{g.r_hat, g.r},
                0.0,
                g.stream
            ));
        }
        allreduce_sum(gpus, {&Shard::d_rho});
        PROF_POP_SYNC(gpus); // step:rho

        // beta - every GPU has the same rho/alpha/omega, so each computes
        // its own copy instead of broadcasting it
        PROF_PUSH("step:beta");
        for (auto& g : gpus){
            CUDA_CHECK(cudaSetDevice(g.dev));
            mult<<<1, 1, 0, g.stream>>>(g.rho_alpha, g.d_rho, g.d_alpha);
            CUDA_CHECK_KERNEL();
            mult<<<1, 1, 0, g.stream>>>(g.rho_p_omega, g.d_rho_p, g.d_omega);
            CUDA_CHECK_KERNEL();
            div<<<1, 1, 0, g.stream>>>(g.d_beta, g.rho_alpha, g.rho_p_omega);
            CUDA_CHECK_KERNEL();
        }
        PROF_POP_SYNC(gpus); // step:beta


        // p
        PROF_PUSH("step:p");
        for (auto& g : gpus){
            CUDA_CHECK(cudaSetDevice(g.dev));
            CUDA_CHECK(cub::DeviceTransform::Transform(
                first,
                g.p + HALO,
                g.n_local,
                p_calc{g.r, g.p + HALO, g.v, g.d_beta, g.d_omega},
                g.stream
            ));
        }
        PROF_POP_SYNC(gpus); // step:p

        // v - needs the neighbours' edge rows of p first
        PROF_PUSH("step:v = A*p");
        PROF_PUSH("comm:halo p");
        halo_exchange(gpus, &Shard::p);
        PROF_POP_SYNC(gpus); // comm:halo p
        for (auto& g : gpus){
            CUDA_CHECK(cudaSetDevice(g.dev));
            CUDA_CHECK(cub::DeviceTransform::Transform(
                first,
                g.v,
                g.n_local,
                v_calc{g.A, g.p, g.row0, n},
                g.stream
            ));
        }
        PROF_POP_SYNC(gpus); // step:v = A*p

        // alpha = rho / (r_hat.v summed over every GPU)

        // alpha
        PROF_PUSH("step:alpha");
        for (auto& g : gpus){
            CUDA_CHECK(cudaSetDevice(g.dev));
            CUDA_CHECK(cub::DeviceReduce::TransformReduce(
                g.d_temp_storage,
                g.temp_bytes,
                first,
                g.r_hat_v,
                g.n_local,
                cuda::std::plus<double>{},
                multiply{g.r_hat, g.v},
                0.0,
                g.stream
            ));
        }
        allreduce_sum(gpus, {&Shard::r_hat_v});
        for (auto& g : gpus){
            CUDA_CHECK(cudaSetDevice(g.dev));
            div<<<1, 1, 0, g.stream>>>(g.d_alpha, g.d_rho, g.r_hat_v);
            CUDA_CHECK_KERNEL();
        }
        PROF_POP_SYNC(gpus); // step:alpha


        // s
        PROF_PUSH("step:s");
        for (auto& g : gpus){
            CUDA_CHECK(cudaSetDevice(g.dev));
            CUDA_CHECK(cub::DeviceTransform::Transform(
                first,
                g.s + HALO,
                g.n_local,
                s_calc{g.r, g.v, g.d_alpha},
                g.stream
            ));
        }
        PROF_POP_SYNC(gpus); // step:s

        // t - needs the neighbours' edge rows of s first
        PROF_PUSH("step:t = A*s");
        PROF_PUSH("comm:halo s");
        halo_exchange(gpus, &Shard::s);
        PROF_POP_SYNC(gpus); // comm:halo s
        for (auto& g : gpus){
            CUDA_CHECK(cudaSetDevice(g.dev));
            CUDA_CHECK(cub::DeviceTransform::Transform(
                first,
                g.t,
                g.n_local,
                t_calc{g.A, g.s, g.row0, n},
                g.stream
            ));
        }
        PROF_POP_SYNC(gpus); // step:t = A*s

        // sTt, tTt - both partials go out in a single allreduce group
        PROF_PUSH("step:omega");
        for (auto& g : gpus){
            CUDA_CHECK(cudaSetDevice(g.dev));
            // sTt
            CUDA_CHECK(cub::DeviceReduce::TransformReduce(
                g.d_temp_storage,
                g.temp_bytes,
                first,
                g.sTt,
                g.n_local,
                cuda::std::plus<double>{},
                multiply{g.s + HALO, g.t},
                0.0,
                g.stream
            ));

            // tTt
            CUDA_CHECK(cub::DeviceReduce::TransformReduce(
                g.d_temp_storage,
                g.temp_bytes,
                first,
                g.tTt,
                g.n_local,
                cuda::std::plus<double>{},
                multiply{g.t, g.t},
                0.0,
                g.stream
            ));
        }
        allreduce_sum(gpus, {&Shard::sTt, &Shard::tTt});

        // omega
        for (auto& g : gpus){
            CUDA_CHECK(cudaSetDevice(g.dev));
            div<<<1, 1, 0, g.stream>>>(g.d_omega, g.sTt, g.tTt);
            CUDA_CHECK_KERNEL();
        }
        PROF_POP_SYNC(gpus); // step:omega

        // update x
        PROF_PUSH("step:x");
        for (auto& g : gpus){
            CUDA_CHECK(cudaSetDevice(g.dev));
            CUDA_CHECK(cub::DeviceTransform::Transform(
                first,
                g.x + HALO,
                g.n_local,
                x_calc{g.x + HALO, g.p + HALO, g.s + HALO, g.d_alpha, g.d_omega},
                g.stream
            ));
        }
        PROF_POP_SYNC(gpus); // step:x

        // update r
        PROF_PUSH("step:r");
        for (auto& g : gpus){
            CUDA_CHECK(cudaSetDevice(g.dev));
            CUDA_CHECK(cub::DeviceTransform::Transform(
                first,
                g.r,
                g.n_local,
                r_calc_update{g.s + HALO, g.t, g.d_omega},
                g.stream
            ));
        }
        PROF_POP_SYNC(gpus); // step:r

        PROF_POP_SYNC(gpus); // iter
    }
    PROF_POP(); // solve:loop
    PROF_PUSH("final:residual_norm");
    std::cout << "residual: " << residual_norm(gpus) << std::endl;
    PROF_POP_SYNC(gpus); // final:residual_norm

    // Nothing here frees itself: every buffer is a raw cudaMalloc on a
    // specific device, so free each GPU's with that GPU current.
    PROF_PUSH("teardown:device_free");
    for (auto& g : gpus){
        CUDA_CHECK(cudaSetDevice(g.dev));
        CUDA_CHECK(cudaFree(g.A));
        CUDA_CHECK(cudaFree(g.b));
        CUDA_CHECK(cudaFree(g.x));
        CUDA_CHECK(cudaFree(g.r));
        CUDA_CHECK(cudaFree(g.r_hat));
        CUDA_CHECK(cudaFree(g.v));
        CUDA_CHECK(cudaFree(g.p));
        CUDA_CHECK(cudaFree(g.s));
        CUDA_CHECK(cudaFree(g.t));

        CUDA_CHECK(cudaFree(g.d_rho));
        CUDA_CHECK(cudaFree(g.d_rho_p));
        CUDA_CHECK(cudaFree(g.d_alpha));
        CUDA_CHECK(cudaFree(g.d_omega));
        CUDA_CHECK(cudaFree(g.d_beta));

        CUDA_CHECK(cudaFree(g.rho_alpha));
        CUDA_CHECK(cudaFree(g.rho_p_omega));
        CUDA_CHECK(cudaFree(g.r_hat_v));
        CUDA_CHECK(cudaFree(g.sTt));
        CUDA_CHECK(cudaFree(g.tTt));
        CUDA_CHECK(cudaFree(g.rTr));

        CUDA_CHECK(cudaFree(g.d_temp_storage));
    }
    PROF_POP_SYNC(gpus); // teardown:device_free

    // Communicators and streams go last, after all work on them has drained.
    PROF_PUSH("teardown:nccl_destroy");
    for (auto& g : gpus){
        CUDA_CHECK(cudaSetDevice(g.dev));
        NCCL_CHECK(ncclCommDestroy(g.comm));
        CUDA_CHECK(cudaStreamDestroy(g.stream));
    }
    PROF_POP(); // teardown:nccl_destroy

    return 0;
}
