#include <chrono>
#include <cstring>
#include <functional>
#include <immintrin.h>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <ratio>

#include "Assignment1_GradeBot.h"

#if defined(BLAS_OPENBLAS)
#include "cblas.h"
#elif defined(BLAS_BLIS)
#include "blis.h"
#include "cblas.h"
#elif defined(BLAS_MKL)
#include "mkl.h"
#include "mkl_cblas.h"
#else
//static_assert(false, "No cblas provider defined");
#endif

#include "fmt/format.h"
#include "matrixMultiply.h"

namespace benchmark {
// Column Primary Only
inline int idx(int size, int col, int row) {
  return col * size + row;
}

double time(std::string tag, std::function<void()> func) {
  auto start_time = std::chrono::high_resolution_clock::now();
  func();
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end_time - start_time;
  std::cout << fmt::format("[{0:}]: {1:}ms", tag, duration.count())
            << std::endl;
  return duration.count();
}

std::unique_ptr<floatType[]> generate_matrix(int size) {
  auto raw_ptr = new floatType[size * size];
  std::unique_ptr<floatType[]> smart_ptr(raw_ptr);
  return smart_ptr;
}

std::unique_ptr<const floatType[]> generate_random_matrix(int size) {
  auto mat_ptr = generate_matrix(size);
  const auto mat = mat_ptr.get();
  const auto len = size * size;
  thread_local std::mt19937 rng(std::random_device{}());
  thread_local std::uniform_real_distribution<double> dist(0., 1.);
#pragma omp parallel for
  for (int i = 0; i < len; i += 1) {
    #ifdef FLOATTYPE_FLOAT32
    mat[i] = floatType(dist(rng));
    #endif
    #ifdef FLOATTYPE_COMPLEX128
    mat[i] = floatType(dist(rng), dist(rng));
    #endif
  }
  return std::move(mat_ptr);
}

double calc_err(int size, const floatType C0[], const floatType C1[]) {
  double err = 0.;
  auto len = size * size;
  for (int i = 0; i < len; i++) {
    auto diff = C0[i] - C1[i];
    err += SQUARE(diff);
  }
  return err;
}
} // namespace benchmark

int main() {
  std::unique_ptr<const floatType[]> A;
  std::unique_ptr<const floatType[]> B;
  std::unique_ptr<floatType[]> C0;
  std::unique_ptr<floatType[]> C1;
  constexpr int MAT_SIZE = 1024;
  benchmark::time("generate_matrice", [&]() {
    A = benchmark::generate_random_matrix(MAT_SIZE);
    B = benchmark::generate_random_matrix(MAT_SIZE);
    C0 = benchmark::generate_matrix(MAT_SIZE);
    C1 = benchmark::generate_matrix(MAT_SIZE);
  });
  std::string cblas_provider_name=
#if defined(BLAS_OPENBLAS)
  "openblas"
#elif defined(BLAS_BLIS)
  "blis"
#elif defined(BLAS_MKL)
  "mkl"
#endif
  ;
  benchmark::time(cblas_provider_name, [&]() {
    floatType alpha = SetOne;
    floatType beta = SetZero;
    #ifdef FLOATTYPE_COMPLEX128
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, MAT_SIZE, MAT_SIZE,
                MAT_SIZE, &alpha, A.get(), MAT_SIZE, B.get(), MAT_SIZE, &beta,
                C0.get(), MAT_SIZE);
    #endif
  });
  benchmark::time("mygemm", [&]() {
    matrixMultiply(MAT_SIZE, A.get(), B.get(), C1.get(),
                                      nullptr, 0);
  });
  benchmark::time("check err", [&]() {
    auto err = benchmark::calc_err(MAT_SIZE, C0.get(), C1.get());
    std::cout << fmt::format("  - error: {}", err) << std::endl;
    if (err > std::numeric_limits<double>::epsilon()) {
      std::cerr << "error exceeded!" << std::endl;
      std::exit(-1);
    }
  });
}
