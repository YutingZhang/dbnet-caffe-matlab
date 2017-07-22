#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

#include <cmath>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <>
void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasDgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasSgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasDgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) {
  CUBLAS_CHECK(cublasSaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

template <>
void caffe_gpu_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) {
  CUBLAS_CHECK(cublasDaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

void caffe_gpu_memcpy(const size_t N, const void* X, void* Y) {
  if (X != Y) {
    CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));  // NOLINT(caffe/alt_fn)
  }
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float *X) {
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double *X) {
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}

template <>
void caffe_gpu_axpby<float>(const int N, const float alpha, const float* X,
    const float beta, float* Y) {
  caffe_gpu_scal<float>(N, beta, Y);
  caffe_gpu_axpy<float>(N, alpha, X, Y);
}

template <>
void caffe_gpu_axpby<double>(const int N, const double alpha, const double* X,
    const double beta, double* Y) {
  caffe_gpu_scal<double>(N, beta, Y);
  caffe_gpu_axpy<double>(N, alpha, X, Y);
}

template <>
void caffe_gpu_dot<float>(const int n, const float* x, const float* y,
    float* out) {
  CUBLAS_CHECK(cublasSdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_dot<double>(const int n, const double* x, const double* y,
    double * out) {
  CUBLAS_CHECK(cublasDdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_asum<float>(const int n, const float* x, float* y) {
  CUBLAS_CHECK(cublasSasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_asum<double>(const int n, const double* x, double* y) {
  CUBLAS_CHECK(cublasDasum(Caffe::cublas_handle(), n, x, 1, y));
}

template <>
void caffe_gpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  CUBLAS_CHECK(cublasScopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <>
void caffe_gpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  CUBLAS_CHECK(cublasDcopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}

template <typename Dtype>
__global__ void set_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = alpha;
  }
}

template <typename Dtype>
void caffe_gpu_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    CUDA_CHECK(cudaMemset(Y, 0, sizeof(Dtype) * N));  // NOLINT(caffe/alt_fn)
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template void caffe_gpu_set<int>(const int N, const int alpha, int* Y);
template void caffe_gpu_set<float>(const int N, const float alpha, float* Y);
template void caffe_gpu_set<double>(const int N, const double alpha, double* Y);

template <typename Dtype>
__global__ void add_scalar_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] += alpha;
  }
}

template <>
void caffe_gpu_add_scalar(const int N, const float alpha, float* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <>
void caffe_gpu_add_scalar(const int N, const double alpha, double* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <typename Dtype>
__global__ void add_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] + b[index];
  }
}

template <>
void caffe_gpu_add<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_add<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void sub_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] - b[index];
  }
}

template <>
void caffe_gpu_sub<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_sub<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void mul_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] * b[index];
  }
}

template <>
void caffe_gpu_mul<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_mul<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void div_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] / b[index];
  }
}

template <>
void caffe_gpu_div<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_div<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void abs_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = abs(a[index]);
  }
}

template <>
void caffe_gpu_abs<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_abs<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}


template <typename Dtype>
__global__ void exp_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = exp(a[index]);
  }
}

template <>
void caffe_gpu_exp<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_exp<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void log_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = log(a[index]);
  }
}

template <>
void caffe_gpu_log<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_log<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void powx_kernel(const int n, const Dtype* a,
    const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = pow(a[index], alpha);
  }
}

template <>
void caffe_gpu_powx<float>(const int N, const float* a,
    const float alpha, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <>
void caffe_gpu_powx<double>(const int N, const double* a,
    const double alpha, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sign, y[index] = (Dtype(0) < x[index])
                                      - (x[index] < Dtype(0)));
DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sgnbit, y[index] = signbit(x[index]));

void caffe_gpu_rng_uniform(const int n, unsigned int* r) {
  CURAND_CHECK(curandGenerate(Caffe::curand_generator(), r, n));
}

template <>
void caffe_gpu_rng_uniform<float>(const int n, const float a, const float b,
                                  float* r) {
  CURAND_CHECK(curandGenerateUniform(Caffe::curand_generator(), r, n));
  const float range = b - a;
  if (range != static_cast<float>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<float>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_uniform<double>(const int n, const double a, const double b,
                                   double* r) {
  CURAND_CHECK(curandGenerateUniformDouble(Caffe::curand_generator(), r, n));
  const double range = b - a;
  if (range != static_cast<double>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<double>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_gaussian(const int n, const float mu, const float sigma,
                            float* r) {
  CURAND_CHECK(
      curandGenerateNormal(Caffe::curand_generator(), r, n, mu, sigma));
}

template <>
void caffe_gpu_rng_gaussian(const int n, const double mu, const double sigma,
                            double* r) {
  CURAND_CHECK(
      curandGenerateNormalDouble(Caffe::curand_generator(), r, n, mu, sigma));
}

// CUSTOM -------------------

template <typename Dtype>
__global__ void zcaffe_gpu_safeinv_kernel(const int n, const Dtype* in, Dtype* out, const Dtype numerator) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = (in[index] == Dtype(0.)) ? 0: (numerator/in[index]);
  }
}

template <typename Dtype>
void zcaffe_gpu_safeinv( const int n, const Dtype* in, Dtype* out, const Dtype numerator ) {
	zcaffe_gpu_safeinv_kernel<Dtype><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
	      n, in, out, numerator);
}

template void zcaffe_gpu_safeinv<float>( const int n, const float* in, float* out, const float numerator );
template void zcaffe_gpu_safeinv<double>( const int n, const double* in, double* out, const double numerator );

template <typename Dtype>
__global__ void zcaffe_gpu_blockcopy_kernel(const int n,
		const int block_size,  const int in_stride,  const Dtype* in,
		const int out_stride, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    int b = index / block_size;
    int o = index % block_size;
    out[b*out_stride+o] = in[b*in_stride+o];
  }
}

template <typename Dtype>
void zcaffe_gpu_blockcopy( const int num_block,
		const int block_size,  const int in_stride,  const Dtype* in,
		const int out_stride, Dtype* out ) {
	const int n = num_block*block_size;
	zcaffe_gpu_blockcopy_kernel<Dtype><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
	      n, block_size, in_stride, in, out_stride, out);
}

template void zcaffe_gpu_blockcopy<unsigned int>( const int num_block,
		const int block_size,  const int in_stride,  const unsigned int* in,
		const int out_stride, unsigned int* out );
template void zcaffe_gpu_blockcopy<int>( const int num_block,
		const int block_size,  const int in_stride,  const int* in,
		const int out_stride, int* out );
template void zcaffe_gpu_blockcopy<float>( const int num_block,
		const int block_size,  const int in_stride,  const float* in,
		const int out_stride, float* out );
template void zcaffe_gpu_blockcopy<double>( const int num_block,
		const int block_size,  const int in_stride,  const double* in,
		const int out_stride, double* out );

template <typename Dtype>
__global__ void zcaffe_gpu_repmul_kernel(const int n,
		const int block, const Dtype* a, const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index]=a[index%block]*b[index];
  }
}

template <typename Dtype>
__global__ void zcaffe_gpu_repmul_py_kernel(const int n,
		const int block, const Dtype* a, const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index]+=a[index%block]*b[index];
  }
}

template <typename Dtype>
void zcaffe_gpu_repmul( const int block,
		const Dtype* a, const int iter, const Dtype* b, Dtype* y, bool is_py ) {
	const int n = block*iter;
	if (is_py)
		zcaffe_gpu_repmul_py_kernel<Dtype><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
		      n, block, a, b, y );
	else
		zcaffe_gpu_repmul_kernel<Dtype><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
		      n, block, a, b, y );
}

template void zcaffe_gpu_repmul<float>(
		const int block, const float* a, const int iter, const float* b, float* y, bool is_py );
template void zcaffe_gpu_repmul<double>(
		const int block, const double* a, const int iter, const double* b, double* y, bool is_py );


template <typename Dtype>
__global__ void zcaffe_gpu_blockaxpy_kernel(const int n,
		const int block_size,  const int x_stride, const Dtype alpha, const Dtype* x,
		const int y_stride, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    int b = index / block_size;
    int o = index % block_size;
    y[b*y_stride+o] += alpha*x[b*x_stride+o];
  }
}

template <typename Dtype>
void zcaffe_gpu_blockaxpy( const int num_block,
		const int block_size,  const int x_stride, const Dtype alpha, const Dtype* x,
		const int y_stride, Dtype* y ) {
	const int n = num_block*block_size;
	zcaffe_gpu_blockaxpy_kernel<Dtype><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
	      n, block_size, x_stride, alpha, x, y_stride, y);
}

template void zcaffe_gpu_blockaxpy<float>( const int num_block,
		const int block_size,  const int x_stride, const float alpha, const float* x,
		const int y_stride, float* y );
template void zcaffe_gpu_blockaxpy<double>( const int num_block,
		const int block_size,  const int x_stride, const double alpha, const double* x,
		const int y_stride, double* y );

// Reordering -----------------------------------------

template <int> struct zcaffe_gpu_reorder_func {
	static bool skip( const int* dims ) { return false; }
private:
	static int reorder( const int n, const int* dims );
	static int dim();
};

template<>
struct zcaffe_gpu_reorder_func<zcaffe_reorder::D3S01> {
	static __device__ int reorder( const int n, const int* dims ) {
		int n0, n1, n2;
		n1 = n /dims[2];
		n0 = n1/dims[1];
		n2 = n %dims[2];
		n1 = n1%dims[1];
        // 1 0 2
		return (n1*dims[0]+n0)*dims[2]+n2;
	}
	static bool skip( const int* dims ) { return (dims[0]==1 || dims[1]==1); }
	static int dim() { return 3; };
};

template<>
struct zcaffe_gpu_reorder_func<zcaffe_reorder::D3S12> {
	static __device__ int reorder( const int n, const int* dims ) {
		int n0, n1, n2;
		n1 = n /dims[2];
		n0 = n1/dims[1];
		n2 = n %dims[2];
		n1 = n1%dims[1];
        // 0 2 1
		return (n0*dims[2]+n2)*dims[1]+n1;
	}
	static bool skip( const int* dims ) { return (dims[1]==1 || dims[2]==1); }
	static int dim() { return 3; };
};

template<>
struct zcaffe_gpu_reorder_func<zcaffe_reorder::D3S02> {
	static __device__ int reorder( const int n, const int* dims ) {
		int n0, n1, n2;
		n1 = n /dims[2];
		n0 = n1/dims[1];
		n2 = n %dims[2];
		n1 = n2%dims[1];
        // 2 1 0
		return (n2*dims[1]+n2)*dims[0]+n0;
	}
	static bool skip( const int* dims ) { return (dims[1]==1 || (dims[0]==1 && dims[2]==1)); }
	static int dim() { return 3; };
};


template <typename Dtype, int ReorderType>
__global__ void zcaffe_gpu_reordering_kernel(const int n,
		const int* dims, const Dtype* A, Dtype*B) {
  typedef zcaffe_gpu_reorder_func<ReorderType> reorder_func;
  CUDA_KERNEL_LOOP(index, n) {
	  int j =reorder_func::reorder(index,dims);
	  B[j] = A[index];
  }
}

template <typename Dtype, int ReorderType >
void zcaffe_gpu_reordering( const int* dims, const Dtype* A, Dtype*B ) {

	if (A==B)
		LOG(WARNING) << "zcaffe_gpu_reordering: Do not support in-place reordering";

	typedef zcaffe_gpu_reorder_func<ReorderType> reorder_func;
	const int dimN = reorder_func::dim();
	int n = 1;
	for ( int i=0; i<dimN; ++i )
		n *= dims[i];
	if ( reorder_func::skip( dims ) ) {
		caffe_copy( n, A, B );
	} else {
		thrust::host_vector<int> H(dims,dims+dimN);
		thrust::device_vector<int> D = H;
		zcaffe_gpu_reordering_kernel<Dtype,ReorderType><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
				n, thrust::raw_pointer_cast(D.data()), A, B );
	}
}

#define INSTANTIATE_GPU_REORDERING(n) \
		template void zcaffe_gpu_reordering<float,n>( const int* dims, const float* A, float* B ); \
		template void zcaffe_gpu_reordering<double,n>( const int* dims, const double* A, double* B );

INSTANTIATE_GPU_REORDERING(zcaffe_reorder::D3S01);
INSTANTIATE_GPU_REORDERING(zcaffe_reorder::D3S12);
INSTANTIATE_GPU_REORDERING(zcaffe_reorder::D3S02);


// -----------------------------------------------------

}  // namespace caffe
