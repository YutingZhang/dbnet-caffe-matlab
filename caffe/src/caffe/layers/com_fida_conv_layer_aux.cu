#include "caffe/layers_aux/fida_conv.hpp"

namespace caffe {


template <typename Dtype>
__global__
void reorder_filters_kernel_conv(const int count,
		const Dtype* W, Dtype* W_tilde,
		const int C, const int C_f, const int R, const int C_out, const int K ) {
  CUDA_KERNEL_LOOP(index, count) {
	  int i = index;
	  int k = i%K; i/=K;
	  int c = i%C; i/=C;
	  int c_out = i; //%C_out;
	  int c_f = c%C_f, r = c/C_f;
	  int j = ((c_f*C_out+c_out)*R+r)*K+k;
	  W_tilde[j] = W[index];
  }
}

template <typename Dtype>
__global__
void reorder_filters_kernel_deconv(const int count,
		const Dtype* W, Dtype* W_tilde,
		const int C_f, const int R, const int C_out, const int K ) {
  CUDA_KERNEL_LOOP(index, count) {
	  int i = index;
	  int k = i%K; i/=K;
	  int c_out = i%C_out; i/=C_out;
	  int c = i; //%C;
	  int c_f = c%C_f, r = c/C_f;
	  int j = ((c_f*R+r)*C_out+c_out)*K+k;
	  W_tilde[j] = W[index];
  }
}

template<typename Dtype>
void CommonFidaConvLayerAux<Dtype, mGPU>::reorder_filters( const Dtype* W, Dtype* W_tilde,
		const int C, const int C_f, const int R, const int C_out, const int K,
		const bool isdeconv ) {
	const int count = C_out*C*K;
	if (isdeconv) {
		reorder_filters_kernel_deconv<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>( count,
				W, W_tilde, C_f, R, C_out, K);
	} else {
		reorder_filters_kernel_conv<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>( count,
				W, W_tilde, C, C_f, R, C_out, K);
	}
	CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__
void input_per_channel_mean_kernel_1(const int count,
		const Dtype* bottom_fidelity_data, const Dtype* bottom_fidelity_sum,
		Dtype* normalized_bottom_fidelity_data, const int G ) {
  CUDA_KERNEL_LOOP(index, count) {
	  int m_f = index/G;
	  normalized_bottom_fidelity_data[index] =
			  bottom_fidelity_data[index]/bottom_fidelity_sum[m_f];
  }
}

template <typename Dtype>
__global__
void input_per_channel_mean_kernel_1_prescaled(const int count,
		const Dtype* bottom_fidelity_sum,
		Dtype* normalized_bottom_fidelity_data, const int G ) {
  CUDA_KERNEL_LOOP(index, count) {
	  int m_f = index/G;
	  normalized_bottom_fidelity_data[index] = Dtype(1.)/bottom_fidelity_sum[m_f];
  }
}

template <typename Dtype>
__global__
void input_per_channel_mean_kernel_2(const int count,
		const Dtype* normalized_bottom_fidelity_data, const Dtype* bottom_feature_data,
		Dtype* bottom_normalized_feature_data,
		const int C, const int C_f, const int G ) {
  CUDA_KERNEL_LOOP(index, count) {
	  int i = index;
	  int g = i%G; i/=G;
	  int c = i%C; i/=C;
	  int n = i; //%N;
	  int c_f = c%C_f;
	  int i_f = (n*C_f+c_f)*G+g;
	  bottom_normalized_feature_data[index] = bottom_feature_data[index] *
		normalized_bottom_fidelity_data[i_f];
  }
}

template<typename Dtype>
void CommonFidaConvLayerAux<Dtype, mGPU>::input_per_channel_mean(
		const Dtype* bottom_fidelity_data, const Dtype* bottom_fidelity_sum, Dtype* normalized_bottom_fidelity_data,
		const Dtype* bottom_feature_data, Dtype* bottom_normalized_feature_data,
		const int N, const int C_f, const int G, const int C, const bool pre_scaled ) {

	{
		const int count = N*C_f*G;
		if (pre_scaled) {
			input_per_channel_mean_kernel_1_prescaled<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>( count,
					bottom_fidelity_sum, normalized_bottom_fidelity_data, G);
		} else {
			input_per_channel_mean_kernel_1<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>( count,
					bottom_fidelity_data, bottom_fidelity_sum, normalized_bottom_fidelity_data, G);
		}
		CUDA_POST_KERNEL_CHECK;
	}
	{
		const int count = N*C*G;
		input_per_channel_mean_kernel_2<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>( count,
				normalized_bottom_fidelity_data, bottom_feature_data, bottom_normalized_feature_data,
				C, C_f, G );
		CUDA_POST_KERNEL_CHECK;
	}

}

template <typename Dtype, int MeanRemoval>
struct reordered_whittened_input_kernel_internal {
};

template <typename Dtype>
struct reordered_whittened_input_kernel_internal<Dtype,0> {
	__device__
	static Dtype func( const Dtype* , const int , const int , const int ) {
		return Dtype(0.);
	}
};

template <typename Dtype>
struct reordered_whittened_input_kernel_internal<Dtype,1> {
	__device__
	static Dtype func( const Dtype* X_bar, const int n, const int C, const int c ) {
		return X_bar[n*C+c];
	}
};


template <typename Dtype, int MeanRemoval>
__global__
void reordered_whittened_input_kernel(const int count,
		const Dtype* X, const Dtype* X_bar, const Dtype* Phi, Dtype* Z,
		const int N, const int C, const int C_f, const int G, const int R ) {
  CUDA_KERNEL_LOOP(index, count) {
	  int i = index;
	  int g = i%G; i/=G;
	  int c = i%C; i/=C;
	  int n = i; //%N

	  int c_f = c%C_f, r = c/C_f;
	  int i_f = (n*C_f+c_f)*G+g;
	  int j   = ((c_f*N+n)*R+r)*G+g;

	  Z[j] = (X[index]-reordered_whittened_input_kernel_internal<Dtype, MeanRemoval>::
			  func(X_bar,n,C,c))*Phi[i_f];
  }
}

template <typename Dtype, int MeanRemoval>
__global__
void reordered_whittened_input_kernel_prescaled(const int count,
		const Dtype* X, const Dtype* X_bar, Dtype* Z,
		const int N, const int C, const int C_f, const int G, const int R ) {
  CUDA_KERNEL_LOOP(index, count) {
	  int i = index;
	  int g = i%G; i/=G;
	  int c = i%C; i/=C;
	  int n = i; //%N

	  int c_f = c%C_f, r = c/C_f;
	  int j   = ((c_f*N+n)*R+r)*G+g;

	  Z[j] = (X[index]-reordered_whittened_input_kernel_internal<Dtype, MeanRemoval>::
			  func(X_bar,n,C,c));
  }
}

template<typename Dtype>
void CommonFidaConvLayerAux<Dtype, mGPU>::reordered_whittened_input(
		const Dtype* X, const Dtype* X_bar, const Dtype* Phi, Dtype* Z,
		const int N, const int C, const int C_f, const int G, const int R,
		const bool mean_removal) {

	const int count = N*C*G;

	if (Phi) {
		if (mean_removal) {
			reordered_whittened_input_kernel<Dtype,1><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>( count,
					X, X_bar, Phi, Z, N, C, C_f, G, R);
		} else {
			reordered_whittened_input_kernel<Dtype,0><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>( count,
					X, X_bar, Phi, Z, N, C, C_f, G, R);
		}
	} else {
		if (mean_removal) {
			reordered_whittened_input_kernel_prescaled<Dtype,1><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>( count,
					X, X_bar, Z, N, C, C_f, G, R);
		} else {
			reordered_whittened_input_kernel_prescaled<Dtype,0><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>( count,
					X, X_bar, Z, N, C, C_f, G, R);
		}
	}
	CUDA_POST_KERNEL_CHECK;

}

template <typename Dtype>
__global__
void forward_backscaling_kernel(const int count,
		const Dtype* YsupR, const Dtype* Omega, Dtype* Y,
		const int C_out, const int G_out, const int C_f, const int c_f) {
  CUDA_KERNEL_LOOP(index, count) {
	  int i = index;
	  int g_out = i%G_out; i/=G_out;
	  //int c_out = i%C_out;
	  i/=C_out;
	  int n = i; //%N
	  int j = (n*C_f+c_f)*G_out+g_out;
	  Y[index]+=YsupR[index]*Omega[j];
  }
}

template<typename Dtype>
void CommonFidaConvLayerAux<Dtype, mGPU>::forward_backscaling(
		const Dtype* YsupR, const Dtype* Omega, Dtype* Y,
		const int N, const int C_out, const int G_out, const int C_f, const int c_f ) {
	const int count = N*C_out*G_out;
	forward_backscaling_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>( count,
			YsupR, Omega, Y, C_out, G_out, C_f, c_f );
	CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__
void backward_scaling_kernel(const int count,
		const Dtype* y_diff, const Dtype* Omega, Dtype* YsupR_diff,
		const int C_out, const int G_out, const int C_f, const int c_f  ) {
  CUDA_KERNEL_LOOP(index, count) {
	  int i = index;
	  int g_out = i%G_out; i/=G_out;
	  //int c_out = i%C_out;
	  i/=C_out;
	  int n = i; //%N
	  int j = (n*C_f+c_f)*G_out+g_out;
	  YsupR_diff[index] = y_diff[index]*Omega[j];
  }
}

template<typename Dtype>
void CommonFidaConvLayerAux<Dtype, mGPU>::backward_scaling(
		const Dtype* y_diff, const Dtype* Omega, Dtype* YsupR_diff,
		const int N, const int C_out, const int G_out, const int C_f, const int c_f ) {
	const int count = N*C_out*G_out;
	backward_scaling_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>( count,
			y_diff, Omega, YsupR_diff, C_out, G_out, C_f, c_f );
	CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype, int MeanAddingBack>
struct filter_diff_kernel_internal {};

template <typename Dtype>
struct filter_diff_kernel_internal<Dtype,0> {
	__device__
	static Dtype func( const Dtype* , const int , const int , const int ) {
		return Dtype(0);
	}
};

template <typename Dtype>
struct filter_diff_kernel_internal<Dtype,1> {
	__device__
	static Dtype func( const Dtype* dh_dw_bar_bar, const int c_out, const int C, const int c ) {
		return dh_dw_bar_bar[c_out*C+c];
	}
};


template <typename Dtype, int MeanAddingBack>
__global__
void filter_diff_kernel_conv(const int count,
		const Dtype* W_tilde_diff, const Dtype* dh_dw_bar_bar, Dtype* W_diff,
		const int C, const int C_f, const int R, const int C_out, const int K ) {
  CUDA_KERNEL_LOOP(index, count) {
	  int i = index;
	  int k = i % K; i/=K;
	  int c = i % C; i/=C;
	  int c_out = i; //% C_out;
	  int c_f = c%C_f, r = c/C_f;
	  int j = ((c_f*C_out+c_out)*R+r)*K+k;
      W_diff[index] += W_tilde_diff[j]+
    		  filter_diff_kernel_internal<Dtype,MeanAddingBack>::func(dh_dw_bar_bar, c_out, C, c);
  }
}

template <typename Dtype, int MeanAddingBack>
__global__
void filter_diff_kernel_deconv(const int count,
		const Dtype* W_tilde_diff, const Dtype* dh_dw_bar_bar, Dtype* W_diff,
		const int C, const int C_f, const int R, const int C_out, const int K ) {
  CUDA_KERNEL_LOOP(index, count) {
	  int i = index;
	  int k = i % K; i/=K;
	  int c_out = i % C_out; i/=C_out;
	  int c = i; //% C;
	  int c_f = c%C_f, r = c/C_f;
	  int j = ((c_f*R+r)*C_out+c_out)*K+k;
      W_diff[index] += W_tilde_diff[j]+
    		  filter_diff_kernel_internal<Dtype,MeanAddingBack>::func(dh_dw_bar_bar, c_out, C, c);
  }
}


template<typename Dtype>
void CommonFidaConvLayerAux<Dtype, mGPU>::filter_diff(
		const Dtype* W_tilde_diff, const Dtype* dh_dw_bar_bar, Dtype* W_diff,
		const int C, const int C_f, const int R, const int C_out, const int K, const int N,
		const bool isdeconv, const bool mean_adding_back ) {

	const int count = C_out*C*K;
	if (isdeconv) {
		if (mean_adding_back)
			filter_diff_kernel_deconv<Dtype,1><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>( count,
					W_tilde_diff, dh_dw_bar_bar, W_diff, C, C_f, R, C_out, K);
		else
			filter_diff_kernel_deconv<Dtype,0><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>( count,
					W_tilde_diff, dh_dw_bar_bar, W_diff, C, C_f, R, C_out, K);
	} else {
		if (mean_adding_back)
			filter_diff_kernel_conv<Dtype,1><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>( count,
					W_tilde_diff, dh_dw_bar_bar, W_diff, C, C_f, R, C_out, K);
		else
			filter_diff_kernel_conv<Dtype,0><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>( count,
					W_tilde_diff, dh_dw_bar_bar, W_diff, C, C_f, R, C_out, K);
	}
	CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype, int PreScaled>
struct input_diff_scaling_kernel_internal {};

template <typename Dtype>
__global__
void input_diff_scaling_kernel(const int count,
		const Dtype* Z_diff, const Dtype* Phi, Dtype* X_diff,
		const int N, const int C, const int C_f, const int G, const int R ) {
  CUDA_KERNEL_LOOP(index, count) {
	  int i = index;
	  int g = i%G; i/=G;
	  int c = i%C; i/=C;
	  int n = i; //%N

	  int c_f = c%C_f, r = c/C_f;
	  int i_f = (n*C_f+c_f)*G+g;
	  int j   = ((c_f*N+n)*R+r)*G+g;

	  X_diff[index] = Z_diff[j]*Phi[i_f];
  }
}

template <typename Dtype>
__global__
void input_diff_scaling_kernel_prescaled(const int count,
		const Dtype* Z_diff, Dtype* X_diff,
		const int N, const int C, const int C_f, const int G, const int R ) {
  CUDA_KERNEL_LOOP(index, count) {
	  int i = index;
	  int g = i%G; i/=G;
	  int c = i%C; i/=C;
	  int n = i; //%N

	  int c_f = c%C_f, r = c/C_f;
	  int j   = ((c_f*N+n)*R+r)*G+g;

	  X_diff[index] = Z_diff[j];
  }
}


template<typename Dtype>
void CommonFidaConvLayerAux<Dtype, mGPU>::input_diff_scaling(
		const Dtype* Z_diff, const Dtype* Phi, Dtype* X_diff,
		const int N, const int C, const int C_f, const int G, const int R ) {
	const int count = N*C*G;
	if (Phi) {
		input_diff_scaling_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>( count,
				Z_diff, Phi, X_diff, N, C, C_f, G, R );
	} else {
		input_diff_scaling_kernel_prescaled<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>( count,
				Z_diff, X_diff, N, C, C_f, G, R );
	}
	CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__
void input_diff_bar_kernel(const int count,
		const Dtype* dh_dX_bar, const Dtype* normalized_bottom_fidelity_data, Dtype* X_diff,
		const int N, const int C, const int C_f, const int G ) {
  CUDA_KERNEL_LOOP(index, count) {
	  int i = index;
	  int g = i%G; i/=G;
	  int c = i%C; i/=C;
	  int n = i; //%N

	  int c_f = c%C_f;
	  int m   = (n*C+c);
	  int i_f = (n*C_f+c_f)*G+g;

	  X_diff[index] += dh_dX_bar[m]*normalized_bottom_fidelity_data[i_f];
  }
}


template<typename Dtype>
void CommonFidaConvLayerAux<Dtype, mGPU>::input_diff_bar(
		const Dtype* dh_dX_bar, const Dtype* normalized_bottom_fidelity_data, Dtype* X_diff,
		const int N, const int C, const int C_f, const int G ) {
	const int count = N*C*G;
	input_diff_bar_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>( count,
			dh_dX_bar, normalized_bottom_fidelity_data, X_diff,
			N, C, C_f, G );
	CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__
void input_fidelity_diff_scaling_kernel(const int count,
		const Dtype* Z_diff, const Dtype* X, Dtype* Phi_diff,
		const int N, const int C, const int C_f, const int G, const int R ) {
  CUDA_KERNEL_LOOP(index, count) {
	  // Phi_diff [N,C_f,G], Z_diff[C_f,N,R,G], X[N,C,G]
	  int i = index;
	  int g = i%G; i/=G;
	  int c_f = i%C_f; i/=C_f;
	  int n = i; //%N

	  int k = (c_f*N+n)*R;
	  int t = n*C;
	  for ( int r=0; r<R; ++r ) {
		  int j = (k+r)*G+g;
		  int c = c_f+r*C_f;
		  int i_im = (t+c)*G+g;
		  Phi_diff[index] += Z_diff[j]*X[i_im];
	  }
  }
}

template<typename Dtype>
void CommonFidaConvLayerAux<Dtype, mGPU>::input_fidelity_diff_scaling(
		const Dtype* Z_diff, const Dtype* X, Dtype* Phi_diff,
		const int N, const int C_f, const int G, const int R ) {
	const int count = N*C_f*G;
	const int C = C_f*R;
	input_fidelity_diff_scaling_kernel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>( count,
			Z_diff, X, Phi_diff, N, C, C_f, G, R );
	CUDA_POST_KERNEL_CHECK;
}


template struct CommonFidaConvLayerAux<float, mGPU>;
template struct CommonFidaConvLayerAux<double, mGPU>;

}
