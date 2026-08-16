#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/memory.h>

#define N 20 // vector/matrix size

int main(void){
    // Initial values - 
    // matrix/vectors
    thrust::device_vector<double> A(N*N);
    thrust::device_vector<double> b(N);
    thrust::device_vector<double> x(N);

    thrust::device_vector<double> r(N);
    thrust::device_vector<double> r_hat(N);

    thrust::device_vector<double> v(N);
    thrust::device_vector<double> p(N);
    
    thrust::device_vector<double> s(N);
    thrust::device_vector<double> t(N);


    // scalars
    double rho, omega, alpha = 1.0f;
    double beta;
    
    // pointers
    double* ptr_A = thrust::raw_pointer_cast(A.data());
    double* ptr_b = thrust::raw_pointer_cast(b.data());
    double* ptr_x = thrust::raw_pointer_cast(x.data());

    // functions
    thrust::counting_iterator<int> first(0);
    auto last = first + N;
    auto r_calc = [=] __device__ (int a){
        double temp = ptr_b[a];
        for (int i = 0; i < 7; i++){
            temp = temp - ptr_A[a*7+i] * ptr_x[a+i];
        }
        return temp;
    };

    // Initial calculations
    thrust::transform(thrust::device, first, last, r.begin(), r_calc);



    return 0;
}