#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/memory.h>
#include <thrust/copy.h>
#include <thrust/inner_product.h>
#include <thrust/fill.h>
#include <thrust/generate.h>
#include <thrust/random.h>
#include <stdlib.h>
#include <stdio.h>
#include <nvtx3/nvToolsExt.h>
#include <iostream>
#include <cub/cub.cuh>
#include <cuda/std/functional>

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

// Kernel launches return void: check the launch, then the execution.
#define CUDA_CHECK_KERNEL()                                                    \
    do {                                                                       \
        CUDA_CHECK(cudaGetLastError());                                        \
        CUDA_CHECK(cudaDeviceSynchronize());                                   \
    } while (0)


// ---- NVTX profiling ranges ----
// Device work is asynchronous, so a range closed straight after a launch would
// time the launch, not the work. PROF_POP_SYNC waits for the GPU first;
// PROF_POP is for host-only regions where there is nothing to wait for.
#define PROF_PUSH(name) nvtxRangePushA(name)
#define PROF_POP()      nvtxRangePop()
#define PROF_POP_SYNC()                                                        \
    do {                                                                       \
        CUDA_CHECK(cudaDeviceSynchronize());                                   \
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


#define ERROR_RATE 1e-8

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

struct r_calc {
    const double* ptr_A;
    const double* ptr_b;
    const double* ptr_x;
    const int n;

    __host__ __device__
    double operator()(int a) const {
        double temp = ptr_b[a];
        for (int i = 0; i < 7; i++){
            int j = a + i - 3;
            if (j < 0 || j >= n) continue;
            temp = temp - ptr_A[a * 7 + i] * ptr_x[j];
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

struct v_calc {
    const double* ptr_A;
    const double* ptr_p;
    const int n;
    
    __host__ __device__
    double operator()(int a) const {
        double temp = 0;
        for (int i = 0; i < 7; i++){
            int j = a + i - 3;
            if (j < 0 || j >= n) continue;
            temp += ptr_A[a * 7 + i] * ptr_p[j];
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

struct t_calc {
    const double* ptr_A;
    const double* ptr_s;
    const int n;

    __host__ __device__
    double operator()(int a) const {
        double temp = 0;
        for (int i = 0; i < 7; i++){
            int j = a + i - 3;
            if (j < 0 || j >= n) continue;
            temp += ptr_A[a * 7 + i] * ptr_s[j];
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

    PROF_POP(); // setup:host_init

    // Initial values - GPU
    // matrix/vectors
    PROF_PUSH("setup:h2d_transfer");
    thrust::device_vector<double> A = h_A;
    thrust::device_vector<double> b = h_b;
    thrust::device_vector<double> x = h_x;
    PROF_POP_SYNC(); // setup:h2d_transfer

    PROF_PUSH("setup:device_alloc");
    thrust::device_vector<double> r(n);
    thrust::device_vector<double> r_hat(n);

    thrust::device_vector<double> v(n); 
    thrust::device_vector<double> p(n); 
    
    thrust::device_vector<double> s(n);
    thrust::device_vector<double> t(n);
    PROF_POP_SYNC(); // setup:device_alloc


    // scalars
    // double rho = 1.0, omega = 1.0, alpha = 1.0;
    // double beta = 0.0, rho_p = 1.0;
    double one = 1.0;
    double zero = 0.0;

    double* d_rho;
    double* d_alpha;
    double* d_omega;
    double* d_beta;
    double* d_rho_p;

    PROF_PUSH("setup:scalar_alloc");
    CUDA_CHECK(cudaMalloc(&d_rho,   sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_rho_p, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_alpha, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_omega, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_beta,  sizeof(double)));
    PROF_POP_SYNC(); // setup:scalar_alloc
    
    PROF_PUSH("setup:scalar_upload");
    CUDA_CHECK(cudaMemcpy(d_rho,   &one, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rho_p,   &one, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_alpha, &one, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_omega, &one, sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta,  &zero, sizeof(double), cudaMemcpyHostToDevice));
    PROF_POP_SYNC(); // setup:scalar_upload
    
    // Intermediary Scalars
    double* rho_alpha;
    double* rho_p_omega;
    double* r_hat_v;
    double* sTt;
    double* tTt;

    PROF_PUSH("setup:scalar_alloc");
    CUDA_CHECK(cudaMalloc(&rho_alpha, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&rho_p_omega, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&r_hat_v, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&sTt, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&tTt, sizeof(double)));
    PROF_POP_SYNC(); // setup:scalar_alloc

    // pointers
    double* ptr_A = thrust::raw_pointer_cast(A.data());
    double* ptr_b = thrust::raw_pointer_cast(b.data());
    double* ptr_x = thrust::raw_pointer_cast(x.data());

    double* ptr_r = thrust::raw_pointer_cast(r.data());
    double* ptr_r_hat = thrust::raw_pointer_cast(r_hat.data());

    double* ptr_p = thrust::raw_pointer_cast(p.data());
    double* ptr_v = thrust::raw_pointer_cast(v.data());

    double* ptr_s = thrust::raw_pointer_cast(s.data());
    double* ptr_t = thrust::raw_pointer_cast(t.data());

    // index sequence the functors are driven by
    thrust::counting_iterator<int> first(0);


    // Initial calculations
    // thrust::transform(thrust::device, first, last, r.begin(), r_calc);
    // thrust::copy(thrust::device, r.begin(), r.end(), r_hat.begin());

    

    PROF_PUSH("init:r0 = b - A*x0");
    CUDA_CHECK(cub::DeviceTransform::Transform(
        first,
        r.begin(),
        r.size(),
        r_calc{ptr_A, ptr_b, ptr_x, n}
    ));
    PROF_POP_SYNC(); // init:r0 = b - A*x0

    PROF_PUSH("init:r_hat = r0");
    thrust::copy(r.begin(), r.end(), r_hat.begin()); // r_hat = r so r.r != 0
    PROF_POP_SYNC(); // init:r_hat = r0

    PROF_PUSH("setup:cub_temp_alloc");
    void* d_temp_storage = nullptr;
    size_t temp_bytes = 0;

    CUDA_CHECK(cub::DeviceReduce::TransformReduce(
        d_temp_storage,
        temp_bytes,
        first,
        d_rho_p,
        r.size(),
        cuda::std::plus<double>{},
        multiply{ptr_r_hat, ptr_r},
        0.0
    ));

    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_bytes));
    PROF_POP_SYNC(); // setup:cub_temp_alloc

    // loop
    PROF_PUSH("init:residual_norm");
    std::cout << thrust::inner_product(thrust::device, r.begin(), r.end(), r.begin(), 0.0) << std::endl;
    PROF_POP_SYNC(); // init:residual_norm
    PROF_PUSH("solve:loop");
    for (int i = 0; i < 50; i++){
        PROF_PUSH("iter");
        // rho_p = rho;
        // rho = thrust::inner_product(thrust::device, r.begin(), r.end(), r_hat.begin(), 0.0); 
        // beta = (rho / rho_p) * (alpha / omega);
        
        PROF_PUSH("step:rho");
        set_value<<<1, 1>>>(d_rho_p, d_rho);
        CUDA_CHECK_KERNEL();

        // rho
        CUDA_CHECK(cub::DeviceReduce::TransformReduce(
            d_temp_storage,
            temp_bytes,
            first,
            d_rho,
            r.size(),
            cuda::std::plus<double>{},
            multiply{ptr_r_hat, ptr_r},
            0.0
        ));
        PROF_POP_SYNC(); // step:rho

        // beta
        PROF_PUSH("step:beta");
        mult<<<1, 1>>>(rho_alpha, d_rho, d_alpha);
        CUDA_CHECK_KERNEL();
        mult<<<1, 1>>>(rho_p_omega, d_rho_p, d_omega);
        CUDA_CHECK_KERNEL();
        div<<<1, 1>>>(d_beta, rho_alpha, rho_p_omega);
        CUDA_CHECK_KERNEL();
        PROF_POP_SYNC(); // step:beta


        // thrust::transform(thrust::device, first, last, p.begin(), p_calc);
        // thrust::transform(thrust::device, first, last, v.begin(), v_calc);

        // p
        PROF_PUSH("step:p");
        CUDA_CHECK(cub::DeviceTransform::Transform(
            first,
            p.begin(),
            p.size(),
            p_calc{ptr_r, ptr_p, ptr_v, d_beta, d_omega}
        ));
        PROF_POP_SYNC(); // step:p

        // v
        PROF_PUSH("step:v = A*p");
        CUDA_CHECK(cub::DeviceTransform::Transform(
            first,
            v.begin(),
            v.size(),
            v_calc{ptr_A, ptr_p, n}
        ));
        PROF_POP_SYNC(); // step:v = A*p

        // alpha = rho/thrust::inner_product(thrust::device, r_hat.begin(), r_hat.end(), v.begin(), 0.0);
        
        // alpha
        PROF_PUSH("step:alpha");
        CUDA_CHECK(cub::DeviceReduce::TransformReduce(
            d_temp_storage,
            temp_bytes,
            first,
            r_hat_v,
            r.size(),
            cuda::std::plus<double>{},
            multiply{ptr_r_hat, ptr_v},
            0.0
        ));
        div<<<1, 1>>>(d_alpha, d_rho, r_hat_v);
        CUDA_CHECK_KERNEL();
        PROF_POP_SYNC(); // step:alpha
        

        // thrust::transform(thrust::device, first, last, s.begin(), s_calc);
        // thrust::transform(thrust::device, first, last, t.begin(), t_calc);

        // s
        PROF_PUSH("step:s");
        CUDA_CHECK(cub::DeviceTransform::Transform(
            first,
            s.begin(),
            s.size(),
            s_calc{ptr_r, ptr_v, d_alpha}
        ));
        PROF_POP_SYNC(); // step:s

        // t
        PROF_PUSH("step:t = A*s");
        CUDA_CHECK(cub::DeviceTransform::Transform(
            first,
            t.begin(),
            t.size(),
            t_calc{ptr_A, ptr_s, n}
        ));
        PROF_POP_SYNC(); // step:t = A*s

        // double st = thrust::inner_product(thrust::device, s.begin(), s.end(), t.begin(), 0.0);
        // double tt = thrust::inner_product(thrust::device, t.begin(), t.end(), t.begin(), 0.0);

        // sTt
        PROF_PUSH("step:omega");
        CUDA_CHECK(cub::DeviceReduce::TransformReduce(
            d_temp_storage,
            temp_bytes,
            first,
            sTt,
            s.size(),
            cuda::std::plus<double>{},
            multiply{ptr_s, ptr_t},
            0.0
        ));

        // tTt
        CUDA_CHECK(cub::DeviceReduce::TransformReduce(
            d_temp_storage,
            temp_bytes,
            first,
            tTt,
            t.size(),
            cuda::std::plus<double>{},
            multiply{ptr_t, ptr_t},
            0.0
        ));
        
        // omega
        div<<<1, 1>>>(d_omega, sTt, tTt);
        CUDA_CHECK_KERNEL();
        PROF_POP_SYNC(); // step:omega

        // thrust::transform(thrust::device, first, last, x.begin(), x_calc);
        // thrust::transform(thrust::device, first, last, r.begin(), r_calc_update);
        
        // update x
        PROF_PUSH("step:x");
        CUDA_CHECK(cub::DeviceTransform::Transform(
            first,
            x.begin(),
            x.size(),
            x_calc{ptr_x, ptr_p, ptr_s, d_alpha, d_omega}
        ));
        PROF_POP_SYNC(); // step:x

        // update r
        PROF_PUSH("step:r");
        CUDA_CHECK(cub::DeviceTransform::Transform(
            first,
            r.begin(),
            r.size(),
            r_calc_update{ptr_s, ptr_t, d_omega}
        ));
        PROF_POP_SYNC(); // step:r


        // std::cout << "rho: " << rho << std::endl;
        // std::cout << "beta: " << beta << std::endl;
        // std::cout << "alpha: " << alpha << std::endl;
        // std::cout << "omega: " << omega << std::endl;
        // std::cout << "rho_p: " << rho_p << std::endl;
        // std::cout << "st: " << st << std::endl;
        // std::cout << "tt: " << tt << std::endl;

        PROF_POP_SYNC(); // iter
    }
    PROF_POP(); // solve:loop
    PROF_PUSH("final:residual_norm");
    std::cout << "residual: " << thrust::inner_product(thrust::device, r.begin(), r.end(), r.begin(), 0.0) << std::endl;
    PROF_POP_SYNC(); // final:residual_norm

    // The device_vectors (A, b, x, r, r_hat, v, p, s, t) release themselves
    // when they go out of scope; the raw cudaMalloc'd scalars do not.
    PROF_PUSH("teardown:device_free");
    CUDA_CHECK(cudaFree(d_rho));
    CUDA_CHECK(cudaFree(d_rho_p));
    CUDA_CHECK(cudaFree(d_alpha));
    CUDA_CHECK(cudaFree(d_omega));
    CUDA_CHECK(cudaFree(d_beta));

    CUDA_CHECK(cudaFree(rho_alpha));
    CUDA_CHECK(cudaFree(rho_p_omega));
    CUDA_CHECK(cudaFree(r_hat_v));
    CUDA_CHECK(cudaFree(sTt));
    CUDA_CHECK(cudaFree(tTt));

    CUDA_CHECK(cudaFree(d_temp_storage));
    PROF_POP_SYNC(); // teardown:device_free

    return 0;
}