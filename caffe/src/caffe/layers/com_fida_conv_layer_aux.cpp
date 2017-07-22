#include "caffe/layers_aux/fida_conv.hpp"

namespace caffe {

template<typename Dtype>
void CommonFidaConvLayerAux<Dtype, mCPU>::reorder_filters( const Dtype* W, Dtype* W_tilde,
		const int C, const int C_f, const int R, const int C_out, const int K,
		const bool isdeconv ) {
	for ( int c_out=0; c_out<C_out; ++c_out ) {
		for ( int c=0; c<C; ++c ) {
			int c_f = c%C_f, r = c/C_f;
			for ( int k=0; k<K; ++k ) {
				int i, j;
				if (isdeconv) {
					i = (c*C_out+c_out)*K+k;			// for W_diff
					j = ((c_f*R+r)*C_out+c_out)*K+k;	// for W_tild_diff
				} else {
					i = (c_out*C+c)*K+k;				// for W_diff
					j = ((c_f*C_out+c_out)*R+r)*K+k;	// for W_tild_diff
				}
				W_tilde[j] = W[i];
			}
		}
	}
}

template<typename Dtype>
void CommonFidaConvLayerAux<Dtype, mCPU>::input_per_channel_mean(
		const Dtype* bottom_fidelity_data, const Dtype* bottom_fidelity_sum, Dtype* normalized_bottom_fidelity_data,
		const Dtype* bottom_feature_data, Dtype* bottom_normalized_feature_data,
		const int N, const int C_f, const int G, const int C, const bool pre_scaled ) {

	for ( int n=0; n<N; ++n ) {
		for  ( int c_f=0; c_f<C_f; ++c_f ) {
			for  ( int g=0; g<G; ++g ) {
				int i_f = (n*C_f+c_f)*G+g;
				int m_f = n*C_f+c_f;
				if ( pre_scaled )
					normalized_bottom_fidelity_data[i_f] =
						Dtype(1.)/bottom_fidelity_sum[m_f];
				else
					normalized_bottom_fidelity_data[i_f] =
						bottom_fidelity_data[i_f]/bottom_fidelity_sum[m_f];
			}
		}
	}
	for ( int n=0; n<N; ++n ) {
		for  ( int c=0; c<C; ++c ) {
			int c_f = c%C_f; //r = c/C_f;
			for  ( int g=0; g<G; ++g ) {
				int i   = (n*C+c)*G+g;
				int i_f = (n*C_f+c_f)*G+g;
				bottom_normalized_feature_data[i] = bottom_feature_data[i] *
						normalized_bottom_fidelity_data[i_f];
			}
		}
	}

}

template<typename Dtype>
void CommonFidaConvLayerAux<Dtype, mCPU>::reordered_whittened_input(
		const Dtype* X, const Dtype* X_bar, const Dtype* Phi, Dtype* Z,
		const int N, const int C, const int C_f, const int G, const int R,
		const bool mean_removal) {

	for ( int n=0; n<N; ++n ) {
		for  ( int c=0; c<C; ++c ) {
			int c_f = c%C_f, r = c/C_f;
			for  ( int g=0; g<G; ++g ) {
				int i   = (n*C+c)*G+g;
				int i_f = (n*C_f+c_f)*G+g;
				int m   = n*C+c;
				int j   = ((c_f*N+n)*R+r)*G+g;
				if (Phi)
					if (mean_removal)
						Z[j] = (X[i]-X_bar[m])*Phi[i_f];
					else
						Z[j] = X[i]*Phi[i_f];
				else
					if (mean_removal)
						Z[j] = (X[i]-X_bar[m]);
					else
						Z[j] = X[i];
			}
		}
	}
}

template<typename Dtype>
void CommonFidaConvLayerAux<Dtype, mCPU>::forward_backscaling(
		const Dtype* YsupR, const Dtype* Omega, Dtype* Y,
		const int N, const int C_out, const int G_out, const int C_f,
		const int c_f ) {
	for ( int n=0; n<N; ++n ) {
		for ( int c_out=0; c_out<C_out; ++c_out ) {
			for ( int g_out=0; g_out<G_out; ++g_out) {
				int i = (n*C_out+c_out)*G_out+g_out;
				int j = (n*C_f+c_f)*G_out+g_out;
				Y[i]+=YsupR[i]*Omega[j];
			}
		}
	}
}


template<typename Dtype>
void CommonFidaConvLayerAux<Dtype, mCPU>::backward_scaling(
		const Dtype* y_diff, const Dtype* Omega, Dtype* YsupR_diff,
		const int N, const int C_out, const int G_out, const int C_f, const int c_f ) {
	for ( int n=0; n<N; ++n ) {
		for ( int c_out=0; c_out<C_out; ++c_out ) {
			for ( int g_out=0; g_out<G_out; ++g_out) {
				int i = (n*C_out+c_out)*G_out+g_out;
				int j = (n*C_f+c_f)*G_out+g_out;
				YsupR_diff[i] = y_diff[i]*Omega[j];
			}
		}
	}
}

template<typename Dtype>
void CommonFidaConvLayerAux<Dtype, mCPU>::filter_diff(
		const Dtype* W_tilde_diff, const Dtype* dh_dw_bar_bar, Dtype* W_diff,
		const int C, const int C_f, const int R, const int C_out, const int K, const int N,
		const bool isdeconv, const bool mean_adding_back ) {

	for ( int c_out=0; c_out<C_out; ++c_out ) {
		for ( int c=0; c<C; ++c ) {
			int c_f = c%C_f, r = c/C_f;
			for ( int k=0; k<K; ++k ) {
				int i, j;
				if (isdeconv) {
					i = (c*C_out+c_out)*K+k;			// for W_diff
					j = ((c_f*R+r)*C_out+c_out)*K+k;	// for W_tild_diff
				} else {
					i = (c_out*C+c)*K+k;				// for W_diff
					j = ((c_f*C_out+c_out)*R+r)*K+k;	// for W_tild_diff
				}
                if (mean_adding_back) {
                    int s = c_out*C+c;			    		// for dh_dw_bar_bar
                    W_diff[i] += W_tilde_diff[j]+dh_dw_bar_bar[s];
                } else {
                    W_diff[i] += W_tilde_diff[j];
                }
			}
		}
	}

}

template<typename Dtype>
void CommonFidaConvLayerAux<Dtype, mCPU>::input_diff_scaling(
		const Dtype* Z_diff, const Dtype* Phi, Dtype* X_diff,
		const int N, const int C, const int C_f, const int G, const int R ) {

	for ( int n=0; n<N; ++n ) {
		for  ( int c=0; c<C; ++c ) {
			int c_f = c%C_f, r = c/C_f;
			for  ( int g=0; g<G; ++g ) {
				int i   = (n*C+c)*G+g;			// for X_diff
				int i_f = (n*C_f+c_f)*G+g;		// for Phi
				int j   = ((c_f*N+n)*R+r)*G+g;	// for Z_diff
				if (Phi)
					X_diff[i] = Z_diff[j]*Phi[i_f];
				else
					X_diff[i] = Z_diff[j];
			}
		}
	}
}

template<typename Dtype>
void CommonFidaConvLayerAux<Dtype, mCPU>::input_diff_bar(
		const Dtype* dh_dX_bar, const Dtype* normalized_bottom_fidelity_data, Dtype* X_diff,
		const int N, const int C, const int C_f, const int G ) {
	for ( int n=0; n<N; ++n ) {
		for  ( int c=0; c<C; ++c ) {
			int c_f = c%C_f; // r = c/C_f;
			for  ( int g=0; g<G; ++g ) {
				int m   = (n*C+c);				// for dh_dX_bar_div_G and X_diff_mean
				int i   = (n*C+c)*G+g;				// for X_diff
				int i_f = (n*C_f+c_f)*G+g;
				X_diff[i] += dh_dX_bar[m]*normalized_bottom_fidelity_data[i_f];
			}
		}
	}
}

template<typename Dtype>
void CommonFidaConvLayerAux<Dtype, mCPU>::input_fidelity_diff_scaling(
		const Dtype* Z_diff, const Dtype* X, Dtype* Phi_diff,
		const int N, const int C_f, const int G, const int R ) {
	int C = C_f*R;
	for ( int n=0; n<N; ++n ) {
		for ( int c_f=0; c_f<C_f; ++c_f ) {
			int k = (c_f*N+n)*R;
			int t = n*C;
			for  ( int g=0; g<G; ++g ) {
                int index = (n*C_f+c_f)*G+g;
				for ( int r=0; r<R; ++r ) {
                      int j = (k+r)*G+g;
					  int c = c_f+r*C_f;
					  int i_im = (t+c)*G+g;
					  Phi_diff[index] += Z_diff[j]*X[i_im];
				}
			}
		}
	}
}

template struct CommonFidaConvLayerAux<float, mCPU>;
template struct CommonFidaConvLayerAux<double, mCPU>;

}
