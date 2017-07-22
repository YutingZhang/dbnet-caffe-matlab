#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <limits>

#include <cstdlib>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include <boost/thread/mutex.hpp>

namespace caffe {

template<>
void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template<>
void caffe_cpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template <>
void caffe_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_cpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) { cblas_saxpy(N, alpha, X, 1, Y, 1); }

template <>
void caffe_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) { cblas_daxpy(N, alpha, X, 1, Y, 1); }

template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

template void caffe_set<int>(const int N, const int alpha, int* Y);
template void caffe_set<float>(const int N, const float alpha, float* Y);
template void caffe_set<double>(const int N, const double alpha, double* Y);

template <>
void caffe_add_scalar(const int N, const float alpha, float* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <>
void caffe_add_scalar(const int N, const double alpha, double* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <typename Dtype>
void caffe_copy(const int N, const Dtype* X, Dtype* Y) {
  if (X != Y) {
    if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      CUDA_CHECK(cudaMemcpy(Y, X, sizeof(Dtype) * N, cudaMemcpyDefault));
#else
      NO_GPU;
#endif
    } else {
      memcpy(Y, X, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    }
  }
}

template void caffe_copy<int>(const int N, const int* X, int* Y);
template void caffe_copy<unsigned int>(const int N, const unsigned int* X,
    unsigned int* Y);
template void caffe_copy<float>(const int N, const float* X, float* Y);
template void caffe_copy<double>(const int N, const double* X, double* Y);

template <>
void caffe_scal<float>(const int N, const float alpha, float *X) {
  cblas_sscal(N, alpha, X, 1);
}

template <>
void caffe_scal<double>(const int N, const double alpha, double *X) {
  cblas_dscal(N, alpha, X, 1);
}

template <>
void caffe_cpu_axpby<float>(const int N, const float alpha, const float* X,
                            const float beta, float* Y) {
  cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_cpu_axpby<double>(const int N, const double alpha, const double* X,
                             const double beta, double* Y) {
  cblas_daxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_add<float>(const int n, const float* a, const float* b,
    float* y) {
  vsAdd(n, a, b, y);
}

template <>
void caffe_add<double>(const int n, const double* a, const double* b,
    double* y) {
  vdAdd(n, a, b, y);
}

template <>
void caffe_sub<float>(const int n, const float* a, const float* b,
    float* y) {
  vsSub(n, a, b, y);
}

template <>
void caffe_sub<double>(const int n, const double* a, const double* b,
    double* y) {
  vdSub(n, a, b, y);
}

template <>
void caffe_mul<float>(const int n, const float* a, const float* b,
    float* y) {
  vsMul(n, a, b, y);
}

template <>
void caffe_mul<double>(const int n, const double* a, const double* b,
    double* y) {
  vdMul(n, a, b, y);
}

template <>
void caffe_div<float>(const int n, const float* a, const float* b,
    float* y) {
  vsDiv(n, a, b, y);
}

template <>
void caffe_div<double>(const int n, const double* a, const double* b,
    double* y) {
  vdDiv(n, a, b, y);
}

template <>
void caffe_powx<float>(const int n, const float* a, const float b,
    float* y) {
  vsPowx(n, a, b, y);
}

template <>
void caffe_powx<double>(const int n, const double* a, const double b,
    double* y) {
  vdPowx(n, a, b, y);
}

template <>
void caffe_sqr<float>(const int n, const float* a, float* y) {
  vsSqr(n, a, y);
}

template <>
void caffe_sqr<double>(const int n, const double* a, double* y) {
  vdSqr(n, a, y);
}

template <>
void caffe_exp<float>(const int n, const float* a, float* y) {
  vsExp(n, a, y);
}

template <>
void caffe_exp<double>(const int n, const double* a, double* y) {
  vdExp(n, a, y);
}

template <>
void caffe_log<float>(const int n, const float* a, float* y) {
  vsLn(n, a, y);
}

template <>
void caffe_log<double>(const int n, const double* a, double* y) {
  vdLn(n, a, y);
}

template <>
void caffe_abs<float>(const int n, const float* a, float* y) {
    vsAbs(n, a, y);
}

template <>
void caffe_abs<double>(const int n, const double* a, double* y) {
    vdAbs(n, a, y);
}

unsigned int caffe_rng_rand() {
  return (*caffe_rng())();
}

template <typename Dtype>
Dtype caffe_nextafter(const Dtype b) {
  return boost::math::nextafter<Dtype>(
      b, std::numeric_limits<Dtype>::max());
}

template
float caffe_nextafter(const float b);

template
double caffe_nextafter(const double b);

template <typename Dtype>
void caffe_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_LE(a, b);
  boost::uniform_real<Dtype> random_distribution(a, caffe_nextafter<Dtype>(b));
  boost::variate_generator<caffe::rng_t*, boost::uniform_real<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_uniform<float>(const int n, const float a, const float b,
                              float* r);

template
void caffe_rng_uniform<double>(const int n, const double a, const double b,
                               double* r);

template <typename Dtype>
void caffe_rng_gaussian(const int n, const Dtype a,
                        const Dtype sigma, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GT(sigma, 0);
  boost::normal_distribution<Dtype> random_distribution(a, sigma);
  boost::variate_generator<caffe::rng_t*, boost::normal_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_gaussian<float>(const int n, const float mu,
                               const float sigma, float* r);

template
void caffe_rng_gaussian<double>(const int n, const double mu,
                                const double sigma, double* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_bernoulli<double>(const int n, const double p, int* r);

template
void caffe_rng_bernoulli<float>(const int n, const float p, int* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, unsigned int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = static_cast<unsigned int>(variate_generator());
  }
}

template
void caffe_rng_bernoulli<double>(const int n, const double p, unsigned int* r);

template
void caffe_rng_bernoulli<float>(const int n, const float p, unsigned int* r);

template <>
float caffe_cpu_strided_dot<float>(const int n, const float* x, const int incx,
    const float* y, const int incy) {
  return cblas_sdot(n, x, incx, y, incy);
}

template <>
double caffe_cpu_strided_dot<double>(const int n, const double* x,
    const int incx, const double* y, const int incy) {
  return cblas_ddot(n, x, incx, y, incy);
}

template <typename Dtype>
Dtype caffe_cpu_dot(const int n, const Dtype* x, const Dtype* y) {
  return caffe_cpu_strided_dot(n, x, 1, y, 1);
}

template
float caffe_cpu_dot<float>(const int n, const float* x, const float* y);

template
double caffe_cpu_dot<double>(const int n, const double* x, const double* y);

template <>
float caffe_cpu_asum<float>(const int n, const float* x) {
  return cblas_sasum(n, x, 1);
}

template <>
double caffe_cpu_asum<double>(const int n, const double* x) {
  return cblas_dasum(n, x, 1);
}

template <>
void caffe_cpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  cblas_scopy(n, x, 1, y, 1);
  cblas_sscal(n, alpha, y, 1);
}

template <>
void caffe_cpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  cblas_dcopy(n, x, 1, y, 1);
  cblas_dscal(n, alpha, y, 1);
}

// CUSTOM -------------------

template <typename Dtype>
void zcaffe_cpu_safeinv( const int n, const Dtype* in, Dtype* out, const Dtype numerator ) {
  for (int i = 0; i < n; ++i) {
	out[i] = (in[i] == Dtype(0))? (Dtype(0)) : (numerator/in[i]);
  }
}

template void zcaffe_cpu_safeinv<float>( const int n, const float* in, float* out, const float numerator );
template void zcaffe_cpu_safeinv<double>( const int n, const double* in, double* out, const double numerator );

template <typename Dtype>
void zcaffe_cpu_blockcopy( const int num_block,
		const int block_size,  const int in_stride,  const Dtype* in,
		const int out_stride, Dtype* out ) {
	for ( int n=0; n<num_block; ++n ) {
		memcpy(out+out_stride*n, in+in_stride*n, sizeof(Dtype)*block_size);
	}
}

template void zcaffe_cpu_blockcopy<unsigned int>( const int num_block,
		const int block_size,  const int in_stride,  const unsigned int* in,
		const int out_stride, unsigned int* out );
template void zcaffe_cpu_blockcopy<int>( const int num_block,
		const int block_size,  const int in_stride,  const int* in,
		const int out_stride, int* out );
template void zcaffe_cpu_blockcopy<float>( const int num_block,
		const int block_size,  const int in_stride,  const float* in,
		const int out_stride, float* out );
template void zcaffe_cpu_blockcopy<double>( const int num_block,
		const int block_size,  const int in_stride,  const double* in,
		const int out_stride, double* out );

template <typename Dtype>
void zcaffe_blockcopy( const int num_block,
		const int block_size,  const int in_stride,  const Dtype* in,
		const int out_stride, Dtype* out ) {
	if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
		zcaffe_gpu_blockcopy( num_block, block_size,
				in_stride, in, out_stride, out );
#else
	      NO_GPU;
#endif
	} else {
		zcaffe_cpu_blockcopy( num_block, block_size,
				in_stride, in, out_stride, out );
	}
}

template void zcaffe_blockcopy<unsigned int>( const int num_block,
		const int block_size,  const int in_stride,  const unsigned int* in,
		const int out_stride, unsigned int* out );
template void zcaffe_blockcopy<int>( const int num_block,
		const int block_size,  const int in_stride,  const int* in,
		const int out_stride, int* out );
template void zcaffe_blockcopy<float>( const int num_block,
		const int block_size,  const int in_stride,  const float* in,
		const int out_stride, float* out );
template void zcaffe_blockcopy<double>( const int num_block,
		const int block_size,  const int in_stride,  const double* in,
		const int out_stride, double* out );


template <typename Dtype>
void zcaffe_cpu_repmul( const int block,
		const Dtype* a, const int iter, const Dtype* b, Dtype* y, bool is_py ) {
	const Dtype* p = b;
    Dtype* q = y;
	if (is_py) {
		for ( int i=0; i<iter; ++i ) {
			const Dtype* t = a;
			for ( int j=0; j<block; ++j )
				*(q++)+=(*t++)*(*p++);
		}
	} else {
		for ( int i=0; i<iter; ++i ) {
			caffe_mul( block, a, p, q );
			p += block; q += block;
		}
	}
}

template void zcaffe_cpu_repmul<float>(
		const int block, const float* a, const int iter, const float* b, float* y, bool is_py );
template void zcaffe_cpu_repmul<double>(
		const int block, const double* a, const int iter, const double* b, double* y, bool is_py );

template <typename Dtype>
void zcaffe_repmul( const int block,
		const Dtype* a, const int iter, const Dtype* b, Dtype* y, bool is_py ) {
	if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
		zcaffe_gpu_repmul( block, a, iter, b, y, is_py );
#else
	      NO_GPU;
#endif
	} else {
		zcaffe_cpu_repmul( block, a, iter, b, y, is_py );
	}
}

template void zcaffe_repmul<float>(
		const int block, const float* a, const int iter, const float* b, float* y, bool is_py );
template void zcaffe_repmul<double>(
		const int block, const double* a, const int iter, const double* b, double* y, bool is_py );


template <typename Dtype>
void zcaffe_cpu_blockaxpy( const int num_block,
		const int block_size,  const int x_stride, const Dtype alpha, const Dtype* x,
		const int y_stride, Dtype* y ) {
	for ( int n=0; n<num_block; ++n ) {
		caffe_axpy( block_size, alpha, x+x_stride*n, y+y_stride*n );
	}
}

template void zcaffe_cpu_blockaxpy<float>( const int num_block,
		const int block_size,  const int x_stride, const float alpha, const float* x,
		const int y_stride, float* y );
template void zcaffe_cpu_blockaxpy<double>( const int num_block,
		const int block_size,  const int x_stride, const double alpha, const double* x,
		const int y_stride, double* y );

template <typename Dtype>
void zcaffe_blockaxpy( const int num_block,
		const int block_size,  const int x_stride, const Dtype alpha, const Dtype* x,
		const int y_stride, Dtype* y ) {
	if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
		zcaffe_gpu_blockaxpy( num_block, block_size, x_stride, alpha, x, y_stride, y );
#else
	      NO_GPU;
#endif
	} else {
		zcaffe_cpu_blockaxpy( num_block, block_size, x_stride, alpha, x, y_stride, y );
	}
}

template void zcaffe_blockaxpy<float>( const int num_block,
		const int block_size,  const int x_stride, const float alpha, const float* x,
		const int y_stride, float* y );
template void zcaffe_blockaxpy<double>( const int num_block,
		const int block_size,  const int x_stride, const double alpha, const double* x,
		const int y_stride, double* y );


template <typename Dtype>
void zcaffe_axpy(const int N, const Dtype alpha, const Dtype* X, Dtype* Y) {
	if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
		caffe_gpu_axpy(N, alpha, X, Y);
#else
	      NO_GPU;
#endif
	} else {
		caffe_axpy(N, alpha, X, Y);
	}
}

template void zcaffe_axpy<float>(const int N, const float alpha, const float* X, float* Y);
template void zcaffe_axpy<double>(const int N, const double alpha, const double* X, double* Y);

template <typename Dtype>
void zcaffe_set(const int N, const Dtype alpha, Dtype* X) {
	if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
		caffe_gpu_set(N, alpha, X);
#else
	      NO_GPU;
#endif
	} else {
		caffe_set(N, alpha, X);
	}
}

template void zcaffe_set<float>(const int N, const float alpha, float* X);
template void zcaffe_set<double>(const int N, const double alpha, double* X);

constexpr double pi_() { return 3.14159265358979323846264338; }
double pi() { return pi_(); };


// Reordering -----------------------------------------

template <int> struct zcaffe_cpu_reorder_func {
	static bool skip( const int* dims ) { return false; }
private:
	static int reorder( const int n, const int* dims );
	static int dim();
};

template<>
struct zcaffe_cpu_reorder_func<zcaffe_reorder::D3S01> {
	static int reorder( const int n, const int* dims ) {
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
struct zcaffe_cpu_reorder_func<zcaffe_reorder::D3S12> {
	static int reorder( const int n, const int* dims ) {
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
struct zcaffe_cpu_reorder_func<zcaffe_reorder::D3S02> {
	static int reorder( const int n, const int* dims ) {
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


template <typename Dtype, int ReorderType >
void zcaffe_cpu_reordering( const int* dims, const Dtype* A, Dtype* B ) {

	if (A==B)
		LOG(WARNING) << "zcaffe_cpu_reordering: Do not support in-place reordering";

	typedef zcaffe_cpu_reorder_func<ReorderType> reorder_func;
    const int dimN = reorder_func::dim();
	int n = 1;
	for ( int i=0; i<dimN; ++i )
		n *= dims[i];
	if ( reorder_func::skip( dims ) ) {
		caffe_copy( n, A, B );
	} else {
		for(int index=0; index<n; ++index ) {
		  int j =reorder_func::reorder(index,dims);
		  B[j]=A[index];
		}
	}
}

#define INSTANTIATE_CPU_REORDERING(n) \
		template void zcaffe_cpu_reordering<float,n>( const int* dims,  const float* A, float* B ); \
		template void zcaffe_cpu_reordering<double,n>( const int* dims, const double* A, double* B );

INSTANTIATE_CPU_REORDERING(zcaffe_reorder::D3S01);
INSTANTIATE_CPU_REORDERING(zcaffe_reorder::D3S12);
INSTANTIATE_CPU_REORDERING(zcaffe_reorder::D3S02);


// -----------------------------------------------------


double rand_threadsafe() {
	static boost::mutex m_;
	boost::mutex::scoped_lock l_(m_);
	return (double)rand()/(double)RAND_MAX;
}

double randn_threadsafe() {
	static boost::mutex m_;
	boost::mutex::scoped_lock l_(m_);

	static boost::mt19937 rng; // I don't seed it on purpouse (it's not relevant)
	static boost::normal_distribution<> nd(0.0, 1.0);
	static boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > var_nor(rng, nd);

	return var_nor();
}

}  // namespace caffe
