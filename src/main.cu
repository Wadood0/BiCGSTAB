#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/memory.h>
#include <thrust/copy.h>
#include <thrust/inner_product.h>
#include <thrust/fill.h>

#define N 20 // vector/matrix size 
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
    // Initial values - 
    // matrix/vectors
    thrust::device_vector<double> A(N*7);
    thrust::device_vector<double> b(N);
    thrust::device_vector<double> x(N);

    thrust::device_vector<double> r(N);
    thrust::device_vector<double> r_hat(N);

    thrust::device_vector<double> v(N); thrust::fill(thrust::device, v.begin(), v.end(), 0.0f);
    thrust::device_vector<double> p(N); thrust::fill(thrust::device, p.begin(), p.end(), 0.0f);
    
    thrust::device_vector<double> s(N);
    thrust::device_vector<double> t(N);


    // scalars
    double rho, omega, alpha = 1.0f;
    double beta, rho_p;
    
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
            temp = temp - ptr_A[a * 7 + i] * ptr_x[a / 7 + i];
        }
        return temp;
    };

    auto p_calc = [=] __device__ (int a){
        return ptr_r[a] + beta * (ptr_p[a] - omega * ptr_v[a]);
    };
    
    auto v_calc = [=] __device__ (int a){
        double temp = 0;
        for (int i = 0; i < 7; i++){
            temp += ptr_A[a * 7 + i] * ptr_p[a+i];
        }
    };

    auto s_calc = [=] __device__ (int a){
        return ptr_r[a] - alpha * ptr_v[a];
    };

    auto t_calc = [=] __device__ (int a){
        double temp = 0;
        for (int i = 0; i < 7; i++){
            temp += ptr_A[a * 7 + i] * ptr_s[a / 7 +i];
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

    // loop
    while (
        thrust::inner_product(thrust::device, r.begin(), r.begin(), r.begin(), 0.0f) > ERROR_RATE
    ){
        rho_p = rho;
        rho = thrust::inner_product(thrust::device, r.begin(), r.begin(), r_hat.begin(), 0.0f); 
        beta = (rho * alpha) / (rho_p * omega); 
        
        thrust::transform(thrust::device, first, last, p.begin(), p_calc);
        thrust::transform(thrust::device, first, last, v.begin(), v_calc);

        alpha = rho/rho_p;

        thrust::transform(thrust::device, first, last, s.begin(), s_calc);
        thrust::transform(thrust::device, first, last, t.begin(), t_calc);

        omega = thrust::inner_product(thrust::device, s.begin(), s.end(), t.begin(), 0.0f)/thrust::inner_product(thrust::device, t.begin(), t.end(), t.begin(), 1.0f);

        thrust::transform(thrust::device, first, last, x.begin(), x_calc);
        thrust::transform(thrust::device, first, last, r.begin(), r_calc);
    }
    return 0;
}