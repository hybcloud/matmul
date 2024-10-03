#include <chrono>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <ratio>

#include "Assignment1_GradeBot.h"
#include "cblas.h"
#include "fmt/format.h"

namespace benchmark {
// Column Primary Only
inline int idx(std::size_t size, std::size_t col, std::size_t row) {
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

std::unique_ptr<floatType[]> generate_matrix(std::size_t size) {
  auto raw_ptr = new floatType[size * size];
  std::unique_ptr<floatType[]> smart_ptr(raw_ptr);
  return smart_ptr;
}

std::unique_ptr<const floatType[]> generate_random_matrix(std::size_t size) {
  auto mat_ptr = generate_matrix(size);
  const auto mat = mat_ptr.get();
  const auto len = size * size;
  thread_local std::mt19937 rng(std::random_device{}());
  thread_local std::uniform_real_distribution<double> dist(0., 1.);
  constexpr int VECTORIZE_LEN = 16;
#pragma omp parallel for
  for (int i = 0; i < len; i += 1) {
    mat[i] = floatType(dist(rng), dist(rng));
  }
  return std::move(mat_ptr);
}

double calc_err(std::size_t size, const floatType C0[], const floatType C1[]) {
  double err = 0.;
  auto len = size * size;
  for (int i = 0; i < len; i++) {
    auto diff = C0[i] - C1[i];
    err += SQUARE(diff);
  }
  return err;
}

namespace mygemm {
using MyGemmFuncType = void (*)(int N, const floatType *A, const floatType *B,
                                floatType *C, int *args, int argCount);
const MyGemmFuncType gemm_brute_force = [](int N, const floatType *A,
                                           const floatType *B, floatType *C,
                                           int *args, int argCount) -> void {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      auto &c = C[idx(N, i, j)];
      c = SetZero;
      for (int k = 0; k < N; k++) {
        c += A[idx(N, k, j)] * B[idx(N, i, k)];
      }
    }
  }
};
const MyGemmFuncType gemm_omp_parallelled =
    [](int N, const floatType *A, const floatType *B, floatType *C, int *args,
       int argCount) -> void {
#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      auto &c = C[idx(N, i, j)];
      c = SetZero;
      for (int k = 0; k < N; k++) {
        c += A[idx(N, k, j)] * B[idx(N, i, k)];
      }
    }
  }
};
const MyGemmFuncType gemm_omp_parallelled_vetorized =
    [](int N, const floatType *A, const floatType *B, floatType *C, int *args,
       int argCount) -> void {
#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      auto &c = C[idx(N, i, j)];
      c = SetZero;
      for (int k = 0; k < N; k++) {
        c += A[idx(N, k, j)] * B[idx(N, i, k)];
      }
    }
  }
};
const MyGemmFuncType gemm_kernelled = [](int N, const floatType *A,
                                         const floatType *B, floatType *C,
                                         int *args, int argCount) -> void {
  constexpr std::size_t KERNEL_SIZE =8;
  thread_local floatType AA[KERNEL_SIZE * KERNEL_SIZE];
  thread_local floatType BB[KERNEL_SIZE * KERNEL_SIZE];
  thread_local floatType CC[KERNEL_SIZE * KERNEL_SIZE];
      for (int k = 0; k < N; k += KERNEL_SIZE) {
  #pragma omp parallel for
  for (int i = 0; i < N; i += KERNEL_SIZE) {
    for (int j = 0; j < N; j += KERNEL_SIZE) {
        // transfer A to kernel AA
        for (int kk = 0; kk < KERNEL_SIZE; kk++) {
          for (int jj = 0; jj < KERNEL_SIZE; jj++) {
            AA[idx(KERNEL_SIZE, kk, jj)] = A[idx(N, k + kk, j + jj)];
          }
        }
        // transfer B to kernel BB
        for (int ii = 0; ii < KERNEL_SIZE; ii++) {
          for (int kk = 0; kk < KERNEL_SIZE; kk++) {
            BB[idx(KERNEL_SIZE, ii, kk)] = B[idx(N, i + ii, k + kk)];
          }
        }
        // init an empty CC
        for (int ii = 0; ii < KERNEL_SIZE; ii++) {
          for (int jj = 0; jj < KERNEL_SIZE; jj++) {
            CC[idx(KERNEL_SIZE, ii, jj)] = 0;
          }
        }
        for (int ii = 0; ii < KERNEL_SIZE; ii++) {
          for (int jj = 0; jj < KERNEL_SIZE; jj++) {
            for (int kk = 0; kk < KERNEL_SIZE; kk++) {
              CC[idx(KERNEL_SIZE, ii, jj)] +=
                  AA[idx(KERNEL_SIZE, kk, jj)] * BB[idx(KERNEL_SIZE, ii, kk)];
            }
          }
        }
        for (int ii = 0; ii < KERNEL_SIZE; ii++) {
          for (int jj = 0; jj < KERNEL_SIZE; jj++) {
            C[idx(N, i + ii, j + jj)] += CC[idx(KERNEL_SIZE, ii, jj)];
          }
        }
        // c += A[idx(N, k, j)] * B[idx(N, i, k)];
      }
    }
  }
};
} // namespace mygemm

} // namespace benchmark

int main() {
  std::unique_ptr<const floatType[]> A;
  std::unique_ptr<const floatType[]> B;
  std::unique_ptr<floatType[]> C0;
  std::unique_ptr<floatType[]> C1;
  constexpr std::size_t MAT_SIZE = 1024;
  benchmark::time("generate_matrice", [&]() {
    A = benchmark::generate_random_matrix(MAT_SIZE);
    B = benchmark::generate_random_matrix(MAT_SIZE);
    C0 = benchmark::generate_matrix(MAT_SIZE);
    C1 = benchmark::generate_matrix(MAT_SIZE);
  });
  benchmark::time("matmul by openblas", [&]() {
    floatType alpha = SetOne;
    floatType beta = SetZero;
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, MAT_SIZE, MAT_SIZE,
                MAT_SIZE, &alpha, A.get(), MAT_SIZE, B.get(), MAT_SIZE, &beta,
                C0.get(), MAT_SIZE);
  });
  benchmark::time("matmul by myblas", [&]() {
    benchmark::mygemm::gemm_omp_parallelled_vetorized(MAT_SIZE, A.get(), B.get(), C1.get(),
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