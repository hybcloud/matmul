//For debugging purposes, you can disable CPU, GPU, and/or MPI sections of the code by commenting out the ENABLE_x defines
#define ENABLE_CPU
// #define ENABLE_GPU
// #define ENABLE_MPI

#define FLOATTYPE_COMPLEX128

//complex64
#ifdef FLOATTYPE_COMPLEX128
#include <complex>
#define floatType std::complex<double>
#define xgemm zgemm
#define cublasXgemm cublasZgemm
#define xgeqrf zgeqrf
#define xorgqr zungqr
#define floatTypeMKL MKL_Complex16
#define floatTypeCUDA cuDoubleComplex
#define SetOne floatType{ 1., 0. }
#define SetZero floatType{ 0. }
#define SQUARE(x) (x.real()*x.real()+x.imag()*x.imag())
#endif