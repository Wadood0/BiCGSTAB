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
#include <cub/cub.cuh>

#define N 100 // vector/matrix size 
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
    float operator()(int i) const
    {
        return a[i] * b[i];
    }
};

struct r_calc {
    const double* ptr_A;
    const double* ptr_b;
    const double* ptr_x;

    __host__ __device__
    double operator()(int a){
        double temp = ptr_b[a];
        for (int i = 0; i < 7; i++){
            temp = temp - ptr_A[a * 7 + i] * ptr_x[a  + i];
        }
        return temp;
    }
}


struct p_calc {
    const double* ptr_r;
    const double* ptr_p;
    const double* ptr_v;
    const double beta;
    const double omega;

    __host__ __device__
    double operator()(int a){
        return ptr_r[a] + beta * (ptr_p[a] - omega * ptr_v[a]);
    }
}

struct v_calc {
    const double* ptr_A;
    const double* ptr_p;
    
    __host__ __device__
    double operator()(int a){
        double temp = 0;
        for (int i = 0; i < 7; i++){
            temp += ptr_A[a * 7 + i] * ptr_p[a  +i];
        }
        return temp;
    }
}

struct s_calc {
    const double* ptr_r;
    const double* ptr_p;
    const double alpha;

    __host__ __device__
    double operator()(int a){
        return ptr_r[a] - alpha * ptr_v[a];
    }
}

struct t_calc {
    const double* ptr_A;
    const double* ptr_s;

    __host__ __device__
    double operator()(int a){
        double temp = 0;
        for (int i = 0; i < 7; i++){
            temp += ptr_A[a * 7 + i] * ptr_s[a +i];
        }
        return temp;
    }

}

struct x_calc {
    const double* ptr_x;
    const double* ptr_p;
    const double* ptr_s;
    const double alpha;
    const double omega;

    __host__ __device__
    double operator()(int a){
        return ptr_x[a] + alpha * ptr_p[a] + omega * ptr_s[a];
    }
}

struct r_calc_update {
    const double* ptr_s;
    const double* ptr_t;
    const double omega;

    __host__ __device__
    double operator()(int a){
        return ptr_s[a] - omega * ptr_t[a];
    }

}

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
    *x = *x * *y;
}

int main(void){

    // Initial values on host to simulate real world environment
    thrust::default_random_engine rng(time(NULL));
    thrust::uniform_real_distribution<double> dist(-1.0, 1.0);

    thrust::host_vector<double> h_A(N*7);
    thrust::host_vector<double> h_b(N);
    thrust::host_vector<double> h_x(N);

    thrust::generate(h_A.begin(), h_A.end(), [&] { return dist(rng); });
    thrust::generate(h_b.begin(), h_b.end(), [&] { return dist(rng); });
    thrust::generate(h_x.begin(), h_x.end(), [&] { return dist(rng); });

    // Zero Padding - IMPORTANT
    int zero_indices_mat[] = {0, 1, 2, 7, 8, 14, N*7-15, N*7-9, N*7-8, N*7-3, N*7-2, N*7-1};
    for (int i = 0; i < 12; i++){
        h_A[zero_indices_mat[i]] = 0.0;
    }


    // Initial values - GPU
    // matrix/vectors
    thrust::device_vector<double> A = h_A;
    thrust::device_vector<double> b = h_b;
    thrust::device_vector<double> x = h_x;

    thrust::device_vector<double> r(N);
    thrust::device_vector<double> r_hat(N);

    thrust::device_vector<double> v(N); 
    thrust::device_vector<double> p(N); 
    
    thrust::device_vector<double> s(N);
    thrust::device_vector<double> t(N);


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

    cudaMalloc(&d_rho,   sizeof(double));
    cudaMalloc(&d_rho_p, sizeof(double));
    cudaMalloc(&d_alpha, sizeof(double));
    cudaMalloc(&d_omega, sizeof(double));
    cudaMalloc(&d_beta,  sizeof(double));
    
    cudaMemcpy(d_rho,   &one, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rho_p,   &one, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_alpha, &one, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_omega, &one, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta,  &zero, sizeof(double), cudaMemcpyHostToDevice);
    
    // Intermediary Scalars
    double* rho_alpha;
    double* rho_p_omega;
    double* r_hat_v;

    // pointers
    double* ptr_A = thrust::raw_pointer_cast(A.data());
    double* ptr_b = thrust::raw_pointer_cast(b.data());
    double* ptr_x = thrust::raw_pointer_cast(x.data());

    double* ptr_r = thrust::raw_pointer_cast(r.data());

    double* ptr_p = thrust::raw_pointer_cast(p.data());
    double* ptr_v = thrust::raw_pointer_cast(v.data());

    double* ptr_s = thrust::raw_pointer_cast(s.data());
    double* ptr_t = thrust::raw_pointer_cast(t.data());


    // Initial calculations
    // thrust::transform(thrust::device, first, last, r.begin(), r_calc);
    // thrust::copy(thrust::device, r.begin(), r.end(), r_hat.begin());

    

    cub::DeviceTransform::Transform(
        first,
        r.begin(),
        r.size(),
        r_calc{ptr_A, ptr_b, ptr_x}
    );

    thrust::copy(r.begin(), r.end(), r_hat()); // r_hat = r so r.r != 0

    void* d_temp_storage = nullptr;
    size_t temp_bytes = 0;

    thrust::DeviceReduce::TransformReduce(
        d_temp_storage,
        temp_bytes,
        first,
        d_rho_p,
        r.size(),
        cuda::std::plus<double>{},
        multiply{r_hat, r},
        0.0
    );

    cudaMalloc(&d_temp_storage, temp_bytes);

    // loop
    std::cout << thrust::inner_product(thrust::device, r.begin(), r.end(), r.begin(), 0.0f) << std::endl;
    for (int i = 0; i < 50; i++){
        // rho_p = rho;
        // rho = thrust::inner_product(thrust::device, r.begin(), r.end(), r_hat.begin(), 0.0f); 
        // beta = (rho / rho_p) * (alpha / omega);
        
        set_value<<<1, 1>>>(d_rho_p, d_rho);

        // rho
        thrust::DeviceReduce::TransformReduce(
            d_temp_storage,
            temp_bytes,
            first,
            d_rho,
            r.size(),
            cuda::std::plus<double>{},
            multiply{r_hat, r},
            0.0
        );

        // beta
        mult<<<1, 1>>>(rho_alpha, d_rho, d_alpha);
        mult<<<1, 1>>>(rho_p_omega, d_rho_p, d_omega);
        div<<<1, 1>>>(d_beta, rho_alpha, rho_p_omega);


        // thrust::transform(thrust::device, first, last, p.begin(), p_calc);
        // thrust::transform(thrust::device, first, last, v.begin(), v_calc);

        // p
        cub::DeviceTransform::Transform(
            first,
            p.begin(),
            p.size(),
            p_calc{ptr_r, ptr_p, ptr_v, d_beta, d_omega}
        );

        // v
        cub::DeviceTransform::Transform(
            first,
            v.begin(),
            v.size(),
            v_calc{ptr_A, ptr_p}
        );

        // alpha = rho/thrust::inner_product(thrust::device, r_hat.begin(), r_hat.end(), v.begin(), 0.0f);
        
        // alpha
        thrust::DeviceReduce::TransformReduce(
            d_temp_storage,
            temp_bytes,
            first,
            r_hat_v,
            r.size(),
            cuda::std::plus<double>{},
            multiply{r_hat, v},
            0.0
        );
        div<<<1, 1>>>(d_alpha, d_rho, r_hat_v);
        




        thrust::transform(thrust::device, first, last, s.begin(), s_calc);
        thrust::transform(thrust::device, first, last, t.begin(), t_calc);

        double st = thrust::inner_product(thrust::device, s.begin(), s.end(), t.begin(), 0.0f);
        double tt = thrust::inner_product(thrust::device, t.begin(), t.end(), t.begin(), 0.0f);

        omega = st / tt;

        thrust::transform(thrust::device, first, last, x.begin(), x_calc);
        thrust::transform(thrust::device, first, last, r.begin(), r_calc_update);

        // std::cout << "rho: " << rho << std::endl;
        // std::cout << "beta: " << beta << std::endl;
        // std::cout << "alpha: " << alpha << std::endl;
        // std::cout << "omega: " << omega << std::endl;
        // std::cout << "rho_p: " << rho_p << std::endl;
        // std::cout << "st: " << st << std::endl;
        // std::cout << "tt: " << tt << std::endl;

    }
    std::cout << "residual: " << thrust::inner_product(thrust::device, r.begin(), r.end(), r.begin(), 0.0f) << std::endl;
    return 0;
}