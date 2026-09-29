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

// Thrust dispatches kernels for us and reports failures by throwing, so there
// is no return code to test. Check the CUDA runtime directly after each call.
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

    CUDA_CHECK_KERNEL(); // allocations + host-to-device copies
    PROF_POP_SYNC(); // setup:device_alloc


    // scalars
    double rho = 1.0, omega = 1.0, alpha = 1.0;
    double beta = 0.0, rho_p = 1.0;
    
    // pointers
    double* ptr_A = thrust::raw_pointer_cast(A.data());
    double* ptr_b = thrust::raw_pointer_cast(b.data());
    double* ptr_x = thrust::raw_pointer_cast(x.data());

    double* ptr_r = thrust::raw_pointer_cast(r.data());

    double* ptr_p = thrust::raw_pointer_cast(p.data());
    double* ptr_v = thrust::raw_pointer_cast(v.data());

    double* ptr_s = thrust::raw_pointer_cast(s.data());
    double* ptr_t = thrust::raw_pointer_cast(t.data());

    // functions
    thrust::counting_iterator<int> first(0);
    auto last = first + n;

    auto r_calc = [=] __device__ (int a){
        double temp = ptr_b[a];
        for (int i = 0; i < 7; i++){
            int j = a + i - 3;
            if (j < 0 || j >= n) continue;
            temp = temp - ptr_A[a * 7 + i] * ptr_x[j];
        }
        return temp;
    };

    auto v_calc = [=] __device__ (int a){
        double temp = 0;
        for (int i = 0; i < 7; i++){
            int j = a + i - 3;
            if (j < 0 || j >= n) continue;
            temp += ptr_A[a * 7 + i] * ptr_p[j];
        }
        return temp;
    };

    auto t_calc = [=] __device__ (int a){
        double temp = 0;
        for (int i = 0; i < 7; i++){
            int j = a + i - 3;
            if (j < 0 || j >= n) continue;
            temp += ptr_A[a * 7 + i] * ptr_s[j];
        }
        return temp;
    };

    // Initial calculations
    PROF_PUSH("init:r0 = b - A*x0");
    thrust::transform(thrust::device, first, last, r.begin(), r_calc);
    CUDA_CHECK_KERNEL();
    PROF_POP_SYNC(); // init:r0 = b - A*x0
    PROF_PUSH("init:r_hat = r0");
    thrust::copy(thrust::device, r.begin(), r.end(), r_hat.begin());
    CUDA_CHECK_KERNEL();
    PROF_POP_SYNC(); // init:r_hat = r0

    // std::cout << rho << std::endl;
    // std::cout << omega << std::endl;
    // std::cout << alpha << std::endl;
    // std::cout << beta << std::endl;

    // loop
    PROF_PUSH("init:residual_norm");
    std::cout << thrust::inner_product(thrust::device, r.begin(), r.end(), r.begin(), 0.0) << std::endl;
    CUDA_CHECK_KERNEL();
    PROF_POP_SYNC(); // init:residual_norm
    
    PROF_PUSH("solve:loop");
    for (int i = 0; i < 50; i++) {
        PROF_PUSH("iter");
        PROF_PUSH("step:rho");
        rho_p = rho;
        rho = thrust::inner_product(thrust::device, r.begin(), r.end(), r_hat.begin(), 0.0); 
        CUDA_CHECK_KERNEL();
        PROF_POP_SYNC(); // step:rho
        PROF_PUSH("step:beta");
        beta = (rho / rho_p) * (alpha / omega);
        PROF_POP(); // step:beta
        

        PROF_PUSH("step:p");
        // __device__ lambdas capture by value, so any lambda that uses a
        // scalar must be rebuilt after that scalar changes.
        auto p_calc = [=] __device__ (int a){
            return ptr_r[a] + beta * (ptr_p[a] - omega * ptr_v[a]);
        };
        thrust::transform(thrust::device, first, last, p.begin(), p_calc);
        CUDA_CHECK_KERNEL();
        PROF_POP_SYNC(); // step:p
        PROF_PUSH("step:v = A*p");
        thrust::transform(thrust::device, first, last, v.begin(), v_calc);
        CUDA_CHECK_KERNEL();
        PROF_POP_SYNC(); // step:v = A*p

        PROF_PUSH("step:alpha");
        alpha = rho/thrust::inner_product(thrust::device, r_hat.begin(), r_hat.end(), v.begin(), 0.0);
        CUDA_CHECK_KERNEL();
        PROF_POP_SYNC(); // step:alpha

        PROF_PUSH("step:s");
        auto s_calc = [=] __device__ (int a){
            return ptr_r[a] - alpha * ptr_v[a];
        };
        thrust::transform(thrust::device, first, last, s.begin(), s_calc);
        CUDA_CHECK_KERNEL();
        PROF_POP_SYNC(); // step:s
        PROF_PUSH("step:t = A*s");
        thrust::transform(thrust::device, first, last, t.begin(), t_calc);
        CUDA_CHECK_KERNEL();
        PROF_POP_SYNC(); // step:t = A*s

        PROF_PUSH("step:omega");
        double st = thrust::inner_product(thrust::device, s.begin(), s.end(), t.begin(), 0.0);
        CUDA_CHECK_KERNEL();
        double tt = thrust::inner_product(thrust::device, t.begin(), t.end(), t.begin(), 0.0);
        CUDA_CHECK_KERNEL();

        omega = st / tt;
        PROF_POP_SYNC(); // step:omega

        PROF_PUSH("step:x");
        auto x_calc = [=] __device__ (int a){
            return ptr_x[a] + alpha * ptr_p[a] + omega * ptr_s[a];
        };
        thrust::transform(thrust::device, first, last, x.begin(), x_calc);
        CUDA_CHECK_KERNEL();
        PROF_POP_SYNC(); // step:x
        PROF_PUSH("step:r");
        auto r_calc_update = [=] __device__ (int a){
            return ptr_s[a] - omega * ptr_t[a];
        };
        thrust::transform(thrust::device, first, last, r.begin(), r_calc_update);
        CUDA_CHECK_KERNEL();
        PROF_POP_SYNC(); // step:r

        PROF_PUSH("io:iteration_print");
        std::cout << "rho: " << rho << std::endl;
        std::cout << "beta: " << beta << std::endl;
        std::cout << "alpha: " << alpha << std::endl;
        std::cout << "omega: " << omega << std::endl;
        std::cout << "rho_p: " << rho_p << std::endl;
        std::cout << "st: " << st << std::endl;
        std::cout << "tt: " << tt << std::endl;
        PROF_POP(); // io:iteration_print

        PROF_POP_SYNC(); // iter
    }
    PROF_POP(); // solve:loop
    PROF_PUSH("final:residual_norm");
    std::cout << "residual: " << thrust::inner_product(thrust::device, r.begin(), r.end(), r.begin(), 0.0) << std::endl;
    CUDA_CHECK_KERNEL();
    PROF_POP_SYNC(); // final:residual_norm
    return 0;
}