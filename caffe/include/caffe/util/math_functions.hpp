#ifndef CAFFE_UTIL_MATH_FUNCTIONS_H_
#define CAFFE_UTIL_MATH_FUNCTIONS_H_

#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit

#include "glog/logging.h"

#include "caffe/common.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/mkl_alternate.hpp"

namespace caffe {

// Caffe gemm provides a simpler interface to the gemm functions, with the
// limitation that the data has to be contiguous in memory.
template <typename Dtype>
void caffe_cpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
    Dtype* C);

template <typename Dtype>
void caffe_cpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
    Dtype* y);

template <typename Dtype>
void caffe_axpy(const int N, const Dtype alpha, const Dtype* X,
    Dtype* Y);

template <typename Dtype>
void caffe_cpu_axpby(const int N, const Dtype alpha, const Dtype* X,
    const Dtype beta, Dtype* Y);

template <typename Dtype>
void caffe_copy(const int N, const Dtype *X, Dtype *Y);

template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype *X);

inline void caffe_memset(const size_t N, const int alpha, void* X) {
  memset(X, alpha, N);  // NOLINT(caffe/alt_fn)
}

template <typename Dtype>
void caffe_add_scalar(const int N, const Dtype alpha, Dtype *X);

template <typename Dtype>
void caffe_scal(const int N, const Dtype alpha, Dtype *X);

template <typename Dtype>
void caffe_sqr(const int N, const Dtype* a, Dtype* y);

template <typename Dtype>
void caffe_add(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_sub(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_mul(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_div(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_powx(const int n, const Dtype* a, const Dtype b, Dtype* y);

unsigned int caffe_rng_rand();

template <typename Dtype>
Dtype caffe_nextafter(const Dtype b);

template <typename Dtype>
void caffe_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r);

template <typename Dtype>
void caffe_rng_gaussian(const int n, const Dtype mu, const Dtype sigma,
                        Dtype* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, int* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, unsigned int* r);

template <typename Dtype>
void caffe_exp(const int n, const Dtype* a, Dtype* y);

template <typename Dtype>
void caffe_log(const int n, const Dtype* a, Dtype* y);

template <typename Dtype>
void caffe_abs(const int n, const Dtype* a, Dtype* y);

template <typename Dtype>
Dtype caffe_cpu_dot(const int n, const Dtype* x, const Dtype* y);

template <typename Dtype>
Dtype caffe_cpu_strided_dot(const int n, const Dtype* x, const int incx,
    const Dtype* y, const int incy);

// Returns the sum of the absolute values of the elements of vector x
template <typename Dtype>
Dtype caffe_cpu_asum(const int n, const Dtype* x);

// the branchless, type-safe version from
// http://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
template<typename Dtype>
inline int8_t caffe_sign(Dtype val) {
  return (Dtype(0) < val) - (val < Dtype(0));
}

// The following two macros are modifications of DEFINE_VSL_UNARY_FUNC
//   in include/caffe/util/mkl_alternate.hpp authored by @Rowland Depp.
// Please refer to commit 7e8ef25c7 of the boost-eigen branch.
// Git cherry picking that commit caused a conflict hard to resolve and
//   copying that file in convenient for code reviewing.
// So they have to be pasted here temporarily.
#define DEFINE_CAFFE_CPU_UNARY_FUNC(name, operation) \
  template<typename Dtype> \
  void caffe_cpu_##name(const int n, const Dtype* x, Dtype* y) { \
    CHECK_GT(n, 0); CHECK(x); CHECK(y); \
    for (int i = 0; i < n; ++i) { \
      operation; \
    } \
  }

// output is 1 for the positives, 0 for zero, and -1 for the negatives
DEFINE_CAFFE_CPU_UNARY_FUNC(sign, y[i] = caffe_sign<Dtype>(x[i]));

// This returns a nonzero value if the input has its sign bit set.
// The name sngbit is meant to avoid conflicts with std::signbit in the macro.
// The extra parens are needed because CUDA < 6.5 defines signbit as a macro,
// and we don't want that to expand here when CUDA headers are also included.
DEFINE_CAFFE_CPU_UNARY_FUNC(sgnbit, \
    y[i] = static_cast<bool>((std::signbit)(x[i])));

DEFINE_CAFFE_CPU_UNARY_FUNC(fabs, y[i] = std::fabs(x[i]));

template <typename Dtype>
void caffe_cpu_scale(const int n, const Dtype alpha, const Dtype *x, Dtype* y);

// --------------------------------------------

double pi();

template <typename Dtype>
void zcaffe_cpu_safeinv( const int n, const Dtype* in, Dtype* out, const Dtype numerator = 1 );

template <typename Dtype>
void zcaffe_cpu_blockcopy( const int num_block,
		const int block_size,  const int in_stride,  const Dtype* in,
		const int out_stride, Dtype* out );

template <typename Dtype>
void zcaffe_blockcopy( const int num_block,
		const int block_size,  const int in_stride,  const Dtype* in,
		const int out_stride, Dtype* out );

template <typename Dtype>
void zcaffe_cpu_repmul( const int block,
		const Dtype* a, const int iter, const Dtype* b, Dtype* y, bool is_py = false );

template <typename Dtype>
void zcaffe_repmul( const int block,
		const Dtype* a, const int iter, const Dtype* b, Dtype* y, bool is_py = false );

template <typename Dtype>
void zcaffe_cpu_blockaxpy( const int num_block,
		const int block_size,  const int x_stride, const Dtype alpha, const Dtype* x,
		const int y_stride, Dtype* y );

template <typename Dtype>
void zcaffe_blockaxpy( const int num_block,
		const int block_size,  const int x_stride, const Dtype alpha, const Dtype* x,
		const int y_stride, Dtype* y );

template <typename Dtype>
void zcaffe_axpy(const int N, const Dtype alpha, const Dtype* X, Dtype* Y);

template <typename Dtype>
void zcaffe_set(const int N, const Dtype alpha, Dtype* X);

template <typename Dtype>
void zcaffe_cpu_swapKM4NKM( const int N, const int K, const int M, Dtype* A );

// ------- Reordering

struct zcaffe_reorder {
	enum {	// row-major
		D3S01,	// 3 dim, swap 1 and 2
		D3S12,	// 3 dim, swap 2 and 3
		D3S02,	// 3 dim, swap 1 and 3
		TOTAL	// total tag
	};
};

template <typename Dtype, int ReorderType >
void zcaffe_cpu_reordering( const int* dims, const Dtype* A, Dtype* B );

#ifndef CPU_ONLY  // GPU

// Decaf gpu gemm provides an interface that is almost the same as the cpu
// gemm function - following the c convention and calling the fortran-order
// gpu code under the hood.
template <typename Dtype>
void caffe_gpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
    Dtype* C);

template <typename Dtype>
void caffe_gpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
    Dtype* y);

template <typename Dtype>
void caffe_gpu_axpy(const int N, const Dtype alpha, const Dtype* X,
    Dtype* Y);

template <typename Dtype>
void caffe_gpu_axpby(const int N, const Dtype alpha, const Dtype* X,
    const Dtype beta, Dtype* Y);

void caffe_gpu_memcpy(const size_t N, const void *X, void *Y);

template <typename Dtype>
void caffe_gpu_set(const int N, const Dtype alpha, Dtype *X);

inline void caffe_gpu_memset(const size_t N, const int alpha, void* X) {
#ifndef CPU_ONLY
  CUDA_CHECK(cudaMemset(X, alpha, N));  // NOLINT(caffe/alt_fn)
#else
  NO_GPU;
#endif
}

template <typename Dtype>
void caffe_gpu_add_scalar(const int N, const Dtype alpha, Dtype *X);

template <typename Dtype>
void caffe_gpu_scal(const int N, const Dtype alpha, Dtype *X);

template <typename Dtype>
void caffe_gpu_add(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_gpu_sub(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_gpu_mul(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_gpu_div(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_gpu_abs(const int n, const Dtype* a, Dtype* y);

template <typename Dtype>
void caffe_gpu_exp(const int n, const Dtype* a, Dtype* y);

template <typename Dtype>
void caffe_gpu_log(const int n, const Dtype* a, Dtype* y);

template <typename Dtype>
void caffe_gpu_powx(const int n, const Dtype* a, const Dtype b, Dtype* y);

// caffe_gpu_rng_uniform with two arguments generates integers in the range
// [0, UINT_MAX].
void caffe_gpu_rng_uniform(const int n, unsigned int* r);

// caffe_gpu_rng_uniform with four arguments generates floats in the range
// (a, b] (strictly greater than a, less than or equal to b) due to the
// specification of curandGenerateUniform.  With a = 0, b = 1, just calls
// curandGenerateUniform; with other limits will shift and scale the outputs
// appropriately after calling curandGenerateUniform.
template <typename Dtype>
void caffe_gpu_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r);

template <typename Dtype>
void caffe_gpu_rng_gaussian(const int n, const Dtype mu, const Dtype sigma,
                            Dtype* r);

template <typename Dtype>
void caffe_gpu_rng_bernoulli(const int n, const Dtype p, int* r);

template <typename Dtype>
void caffe_gpu_dot(const int n, const Dtype* x, const Dtype* y, Dtype* out);

template <typename Dtype>
void caffe_gpu_asum(const int n, const Dtype* x, Dtype* y);

template<typename Dtype>
void caffe_gpu_sign(const int n, const Dtype* x, Dtype* y);

template<typename Dtype>
void caffe_gpu_sgnbit(const int n, const Dtype* x, Dtype* y);

template <typename Dtype>
void caffe_gpu_fabs(const int n, const Dtype* x, Dtype* y);

template <typename Dtype>
void caffe_gpu_scale(const int n, const Dtype alpha, const Dtype *x, Dtype* y);

#define DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(name, operation) \
template<typename Dtype> \
__global__ void name##_kernel(const int n, const Dtype* x, Dtype* y) { \
  CUDA_KERNEL_LOOP(index, n) { \
    operation; \
  } \
} \
template <> \
void caffe_gpu_##name<float>(const int n, const float* x, float* y) { \
  /* NOLINT_NEXT_LINE(whitespace/operators) */ \
  name##_kernel<float><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>( \
      n, x, y); \
} \
template <> \
void caffe_gpu_##name<double>(const int n, const double* x, double* y) { \
  /* NOLINT_NEXT_LINE(whitespace/operators) */ \
  name##_kernel<double><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>( \
      n, x, y); \
}

template <typename Dtype>
void zcaffe_gpu_safeinv( const int n, const Dtype* in, Dtype* out, const Dtype numerator = 1 );

template <typename Dtype>
void zcaffe_gpu_blockcopy( const int num_block,
		const int block_size,  const int in_stride,  const Dtype* in,
		const int out_stride, Dtype* out );

template <typename Dtype>
void zcaffe_gpu_repmul( const int block,
		const Dtype* a, const int iter, const Dtype* b, Dtype* y, bool is_py = false );

template <typename Dtype>
void zcaffe_gpu_blockaxpy( const int num_block,
		const int block_size,  const int x_stride, const Dtype alpha, const Dtype* x,
		const int y_stride, Dtype* y );

template <typename Dtype, int ReorderType >
void zcaffe_gpu_reordering( const int* dims, const Dtype* A, Dtype* B );

#endif  // !CPU_ONLY

// generic caller -----------------------------------------------------------------------------

template<class DeviceMode>
struct gcm {};


template<>
struct gcm<mCPU> {
	template <typename Dtype>
	static void gemm(const CBLAS_TRANSPOSE TransA,
	    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
	    const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
	    Dtype* C) {
		caffe_cpu_gemm(TransA,TransB,M,N,K,alpha,A,B,beta,C);
	}

	template <typename Dtype>
	static void gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
	    const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
	    Dtype* y) {
		caffe_cpu_gemv(TransA,M,N,alpha,A,x,beta,y);
	}

	template <typename Dtype>
	static void set(const int N, const Dtype alpha, Dtype *X) {
		caffe_set(N,alpha,X);
	}

	template<typename Dtype>
	static void safeinv(  const int n, const Dtype* in, Dtype* out, const Dtype numerator = 1 ) {
		zcaffe_cpu_safeinv( n,in,out,numerator );
	}

	template <typename Dtype>
	static void mul(const int N, const Dtype* a, const Dtype* b, Dtype* y) {
		caffe_mul(N, a, b, y);
	}

	template <typename Dtype>
	static void powx(const int n, const Dtype* a, const Dtype b, Dtype* y) {
		caffe_powx(n, a, b, y);
	}

	template <typename Dtype>
	static void axpy(const int N, const Dtype alpha, const Dtype* X, Dtype* Y) {
		caffe_cpu_axpy( N, alpha, X, Y );
	}

	template <typename Dtype>
	static void div(const int N, const Dtype* a, const Dtype* b, Dtype* y) {
		caffe_div(N, a, b, y);
	}

	template <typename Dtype>
	static void add_scalar(const int N, const Dtype alpha, Dtype *X) {
		caffe_add_scalar(N, alpha, X);
	}

	template <typename Dtype>
	static void add(const int N, const Dtype* a, const Dtype* b, Dtype* y) {
		caffe_add(N, a, b, y);
	}

	template <typename Dtype>
	static void sub(const int N, const Dtype* a, const Dtype* b, Dtype* y) {
		caffe_sub(N, a, b, y);
	}

	template <typename Dtype>
	static void axpby(const int N, const Dtype alpha, const Dtype* X,
	    const Dtype beta, Dtype* Y) {
		caffe_cpu_axpby(N, alpha, X, beta, Y);
	}

	template <typename Dtype>
	static void scal(const int N, const Dtype alpha, Dtype *X) {
		caffe_scal(N, alpha, X);
	}


};

#ifndef CPU_ONLY  // GPU

template<>
struct gcm<mGPU> {
	template <typename Dtype>
	static void gemm(const CBLAS_TRANSPOSE TransA,
	    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
	    const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
	    Dtype* C) {
		caffe_gpu_gemm(TransA,TransB,M,N,K,alpha,A,B,beta,C);
	}

	template <typename Dtype>
	static void gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
	    const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
	    Dtype* y) {
		caffe_gpu_gemv(TransA,M,N,alpha,A,x,beta,y);
	}

	template <typename Dtype>
	static void set(const int N, const Dtype alpha, Dtype *X) {
		caffe_gpu_set(N,alpha,X);
	}

	template<typename Dtype>
	static void safeinv(  const int n, const Dtype* in, Dtype* out, const Dtype numerator = 1 ) {
		zcaffe_gpu_safeinv( n,in,out,numerator );
	}

	template <typename Dtype>
	static void mul(const int N, const Dtype* a, const Dtype* b, Dtype* y) {
		caffe_gpu_mul(N, a, b, y);
	}

	template <typename Dtype>
	static void powx(const int n, const Dtype* a, const Dtype b, Dtype* y) {
		caffe_gpu_powx(n, a, b, y);
	}

	template <typename Dtype>
	static void axpy(const int N, const Dtype alpha, const Dtype* X, Dtype* Y) {
		caffe_gpu_axpy( N, alpha, X, Y );
	}

	template <typename Dtype>
	static void div(const int N, const Dtype* a, const Dtype* b, Dtype* y) {
		caffe_gpu_div(N, a, b, y);
	}

	template <typename Dtype>
	static void add_scalar(const int N, const Dtype alpha, Dtype *X) {
		caffe_gpu_add_scalar(N, alpha, X);
	}

	template <typename Dtype>
	static void add(const int N, const Dtype* a, const Dtype* b, Dtype* y) {
		caffe_gpu_add(N, a, b, y);
	}

	template <typename Dtype>
	static void sub(const int N, const Dtype* a, const Dtype* b, Dtype* y) {
		caffe_gpu_sub(N, a, b, y);
	}

	template <typename Dtype>
	static void axpby(const int N, const Dtype alpha, const Dtype* X,
	    const Dtype beta, Dtype* Y) {
		caffe_gpu_axpby(N, alpha, X, beta, Y);
	}

	template <typename Dtype>
	static void scal(const int N, const Dtype alpha, Dtype *X) {
		caffe_gpu_scal(N, alpha, X);
	}


};


#endif

// --------------------------------------------------------------------------------------------


double rand_threadsafe();
double randn_threadsafe();

}  // namespace caffe

#endif  // CAFFE_UTIL_MATH_FUNCTIONS_H_
