#include "matrixMultiply.h"

namespace mygemm {
// Column Primary Only
inline std::size_t idx(std::size_t size, std::size_t col, std::size_t row) {
  return col * size + row;
}

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
  memset(C, 0, sizeof(floatType) * N * N);
#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    for (int k = 0; k < N; k++) {
      for (int j = 0; j < N; j++) {
        C[idx(N, i, j)] += A[idx(N, k, j)] * B[idx(N, i, k)];
      }
    }
  }
};
namespace utils {
namespace gemm_kernelled {
constexpr int KERNEL_SIZE = 16;
floatType AA[KERNEL_SIZE * KERNEL_SIZE];
floatType BB[KERNEL_SIZE * KERNEL_SIZE];
floatType CC[KERNEL_SIZE * KERNEL_SIZE];
#pragma omp threadprivate(AA, BB, CC)
} // namespace gemm_kernelled

} // namespace utils
const MyGemmFuncType gemm_kernelled = [](int N, const floatType *A,
                                         const floatType *B, floatType *C,
                                         int *args, int argCount) -> void {
  using namespace utils::gemm_kernelled;
  #pragma omp parallel for
  for (int i = 0; i < N; i += KERNEL_SIZE) {
    for (int j = 0; j < N; j += KERNEL_SIZE) {
      // init an empty CC
      for (int ii = 0; ii < KERNEL_SIZE; ii++) {
        for (int jj = 0; jj < KERNEL_SIZE; jj++) {
          CC[idx(KERNEL_SIZE, ii, jj)] = SetZero;
        }
      }
      for (int k = 0; k < N; k += KERNEL_SIZE) {
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
        for (int ii = 0; ii < KERNEL_SIZE; ii++) {
          for (int kk = 0; kk < KERNEL_SIZE; kk++) {
            #pragma omp simd
            for (int jj = 0; jj < KERNEL_SIZE; jj++) {
              CC[idx(KERNEL_SIZE, ii, jj)] +=
                  AA[idx(KERNEL_SIZE, kk, jj)] * BB[idx(KERNEL_SIZE, ii, kk)];
            }
          }
        }
        // c += A[idx(N, k, j)] * B[idx(N, i, k)];
      }
      for (int ii = 0; ii < KERNEL_SIZE; ii++) {
        for (int jj = 0; jj < KERNEL_SIZE; jj++) {
          C[idx(N, i + ii, j + jj)] = CC[idx(KERNEL_SIZE, ii, jj)];
        }
      }
    }
  }
};
} // namespace mygemm

void matrixMultiply(int N, const floatType *A, const floatType *B, floatType *C,
                    int *args, int argCount) {
  const mygemm::MyGemmFuncType inner_gemm_func = mygemm::gemm_kernelled;
  inner_gemm_func(N, A, B, C, args, argCount);
}