#include "matrixMultiply.h"
#include <complex>

// #pragma GCC optimize ("Ofast")
namespace mygemm {
// Column Primary Only
inline std::size_t idx(std::size_t size, std::size_t col, std::size_t row) {
  return col * size + row;
}

using MyGemmFuncType = void (*)(int N, const floatType *A, const floatType *B,
                                floatType *C, int *args, int argCount);
const MyGemmFuncType gemm_brute_force = [](int N, const floatType *A,
                                           const floatType *B, floatType *C,
                                           int *, int) -> void {
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
const MyGemmFuncType gemm_ikj_omp_parallelled =
    [](int N, const floatType *A, const floatType *B, floatType *C, int *args,
       int argCount) -> void {
  new (C) floatType[N * N]();
#pragma omp parallel for
  for (int i = 0; i < N; i++) {
    for (int k = 0; k < N; k++) {
      for (int j = 0; j < N; j++) {
        C[idx(N, i, j)] += A[idx(N, k, j)] * B[idx(N, i, k)];
      }
    }
  }
};
const MyGemmFuncType gemm_kij = [](int N, const floatType *A,
                                   const floatType *B, floatType *C, int *,
                                   int) -> void {
  new (C) floatType[N * N]();
  for (int k = 0; k < N; k++) {
    {
      for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
          C[idx(N, i, j)] += A[idx(N, k, j)] * B[idx(N, i, k)];
        }
      }
    }
  }
};

namespace utils {
namespace gemm_blocked {
constexpr int BLOCK_SIZE = 32;
floatType AA[BLOCK_SIZE * BLOCK_SIZE];
floatType BB[BLOCK_SIZE * BLOCK_SIZE];
floatType CC[BLOCK_SIZE * BLOCK_SIZE];
#pragma omp threadprivate(AA, BB, CC)
} // namespace gemm_blocked

} // namespace utils
const MyGemmFuncType gemm_blocked = [](int N, const floatType *A,
                                       const floatType *B, floatType *C, int *,
                                       int) -> void {
  using namespace utils::gemm_blocked;
  new (C) floatType[N * N]();
#pragma omp parallel for
  for (int i = 0; i < N; i += BLOCK_SIZE) {
    for (int j = 0; j < N; j += BLOCK_SIZE) {
      // init an empty CC
      for (int ii = 0; ii < BLOCK_SIZE; ii++) {
        for (int jj = 0; jj < BLOCK_SIZE; jj++) {
          CC[idx(BLOCK_SIZE, ii, jj)] = SetZero;
        }
      }
      for (int k = 0; k < N; k += BLOCK_SIZE) {
        // transfer A to kernel AA
        for (int kk = 0; kk < BLOCK_SIZE; kk++) {
          for (int jj = 0; jj < BLOCK_SIZE; jj++) {
            AA[idx(BLOCK_SIZE, kk, jj)] = A[idx(N, k + kk, j + jj)];
          }
        }
        // transfer B to kernel BB
        for (int ii = 0; ii < BLOCK_SIZE; ii++) {
          for (int kk = 0; kk < BLOCK_SIZE; kk++) {
            BB[idx(BLOCK_SIZE, ii, kk)] = B[idx(N, i + ii, k + kk)];
          }
        }
        for (int ii = 0; ii < BLOCK_SIZE; ii++) {
          for (int kk = 0; kk < BLOCK_SIZE; kk++) {
            // #pragma omp simd
            for (int jj = 0; jj < BLOCK_SIZE; jj++) {
              CC[idx(BLOCK_SIZE, ii, jj)] +=
                  AA[idx(BLOCK_SIZE, kk, jj)] * BB[idx(BLOCK_SIZE, ii, kk)];
            }
          }
        }
        // c += A[idx(N, k, j)] * B[idx(N, i, k)];
      }
      for (int ii = 0; ii < BLOCK_SIZE; ii++) {
        for (int jj = 0; jj < BLOCK_SIZE; jj++) {
          C[idx(N, i + ii, j + jj)] = CC[idx(BLOCK_SIZE, ii, jj)];
        }
      }
    }
  }
};
} // namespace mygemm
namespace mygemm {
namespace utils {
namespace gemm_sota {
void *aligned_malloc(std::size_t alignment, std::size_t size) {
  void *original = malloc(size + alignment - 1 + sizeof(void *));
  if (original == nullptr) {
    return nullptr;
  }
  void *aligned =
      (void *)(((uintptr_t)original + alignment - 1 + sizeof(void *)) &
               ~(alignment - 1));
  ((void **)aligned)[-1] = original;
  return aligned;
}
void aligned_free(void *aligned) {
  if (aligned != NULL) {
    free(((void **)aligned)[-1]);
  }
}
constexpr int MC = 256;
constexpr int KC = 256;
constexpr int NC = 256;
constexpr int MR = 4;
constexpr int NR = 4;
// alignas(64) floatType CC[MR * NR];
floatType AA[KC * MC];
floatType BB[NC * KC];
floatType CC[MR * NR];
#pragma omp threadprivate(AA, BB, CC)
} // namespace gemm_sota
} // namespace utils
MyGemmFuncType gemm_sota = [](int N, const floatType *A, const floatType *B,
                              floatType *C, int *, int) -> void {
  using namespace utils::gemm_sota;

  for (int i = 0; i < N * N; i++) {
    C[i] = floatType(0);
  }
#pragma omp parallel for
  for (int i = 0; i < N; i += NC) {
    for (int k = 0; k < N; k += KC) {
      for (int j = 0; j < N; j += MC) {
        for (int ii = 0; ii < NC; ii += NR) {
          for (int jj = 0; jj < MC; jj += MR) {
            for (int i_kern = 0; i_kern < MR * NR; i_kern++) {
              CC[i_kern] = floatType(0);
            }
            for (int k_kern = 0; k_kern < KC; k_kern++) {
              for (int i_kern = 0; i_kern < NR; i_kern++) {
#pragma omp simd
                for (int j_kern = 0; j_kern < MR; j_kern++) {
                  CC[idx(NR, k_kern, j_kern)] +=
                      A[idx(N, k + k_kern, j + jj + j_kern)] *
                      B[idx(N, i + ii + i_kern, k + k_kern)];
                }
              }
            }
            for (int i_kern = 0; i_kern < NR; i_kern++) {
              for (int j_kern = 0; j_kern < MR; j_kern++) {
                C[idx(N, i + ii + i_kern, j + jj + j_kern)] +=
                    CC[idx(NR, i_kern, j_kern)];
              }
            }
          }
        }
      }
    }
  }
};
} // namespace mygemm
void matrixMultiply(int N, const floatType *A, const floatType *B, floatType *C,
                    int *args, int argCount) {
  const mygemm::MyGemmFuncType inner_gemm_func = mygemm::gemm_sota;
  inner_gemm_func(N, A, B, C, args, argCount);
}