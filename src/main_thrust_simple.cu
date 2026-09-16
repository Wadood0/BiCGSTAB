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
    auto last = first + N;

    auto r_calc = [=] __device__ (int a){
        double temp = ptr_b[a];
        for (int i = 0; i < 7; i++){
            temp = temp - ptr_A[a * 7 + i] * ptr_x[a  + i];
        }
        return temp;
    };

    auto p_calc = [=] __device__ (int a){
        return ptr_r[a] + beta * (ptr_p[a] - omega * ptr_v[a]);
    };
    
    auto v_calc = [=] __device__ (int a){
        double temp = 0;
        for (int i = 0; i < 7; i++){
            temp += ptr_A[a * 7 + i] * ptr_p[a  +i];
        }
        return temp;
    };

    auto s_calc = [=] __device__ (int a){
        return ptr_r[a] - alpha * ptr_v[a];
    };

    auto t_calc = [=] __device__ (int a){
        double temp = 0;
        for (int i = 0; i < 7; i++){
            temp += ptr_A[a * 7 + i] * ptr_s[a +i];
        }
        return temp;
    };

    auto x_calc = [=] __device__ (int a){
        return ptr_x[a] + alpha * ptr_p[a] + omega * ptr_s[a];
    };

    auto r_calc_update = [=] __device__ (int a){
        return ptr_s[a] - omega * ptr_t[a];
    };

    // Initial calculations
    thrust::transform(thrust::device, first, last, r.begin(), r_calc);
    thrust::copy(thrust::device, r.begin(), r.end(), r_hat.begin());

    // std::cout << rho << std::endl;
    // std::cout << omega << std::endl;
    // std::cout << alpha << std::endl;
    // std::cout << beta << std::endl;

    // loop
    std::cout << thrust::inner_product(thrust::device, r.begin(), r.end(), r.begin(), 0.0f) << std::endl;
    
    for (int i = 0; i < 50; i++) {
        rho_p = rho;
        rho = thrust::inner_product(thrust::device, r.begin(), r.end(), r_hat.begin(), 0.0f); 
        beta = (rho / rho_p) * (alpha / omega);
        

        thrust::transform(thrust::device, first, last, p.begin(), p_calc);
        thrust::transform(thrust::device, first, last, v.begin(), v_calc);

        alpha = rho/thrust::inner_product(thrust::device, r_hat.begin(), r_hat.end(), v.begin(), 0.0f);

        thrust::transform(thrust::device, first, last, s.begin(), s_calc);
        thrust::transform(thrust::device, first, last, t.begin(), t_calc);

        double st = thrust::inner_product(thrust::device, s.begin(), s.end(), t.begin(), 0.0f);
        double tt = thrust::inner_product(thrust::device, t.begin(), t.end(), t.begin(), 0.0f);

        omega = st / tt;

        thrust::transform(thrust::device, first, last, x.begin(), x_calc);
        thrust::transform(thrust::device, first, last, r.begin(), r_calc_update);

        std::cout << "rho: " << rho << std::endl;
        std::cout << "beta: " << beta << std::endl;
        std::cout << "alpha: " << alpha << std::endl;
        std::cout << "omega: " << omega << std::endl;
        std::cout << "rho_p: " << rho_p << std::endl;
        std::cout << "st: " << st << std::endl;
        std::cout << "tt: " << tt << std::endl;

    }
    std::cout << "residual: " << thrust::inner_product(thrust::device, r.begin(), r.end(), r.begin(), 0.0f) << std::endl;
    return 0;
}