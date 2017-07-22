#ifndef CAFFE_LAYERS_AUX_FIDA_CONV_HPP_
#define CAFFE_LAYERS_AUX_FIDA_CONV_HPP_

#include "caffe/common.hpp"

namespace caffe {

template<typename Dtype, class DeviceMode>
struct CommonFidaConvLayerAux {};

template<typename Dtype>
struct CommonFidaConvLayerAux<Dtype, mCPU> {
	// Forward
	static void reorder_filters(const Dtype* W, Dtype* W_tilde,
			const int C, const int C_f, const int R, const int C_out, const int K,
			const bool isdeconv );
	static void input_per_channel_mean(
			const Dtype* bottom_fidelity_data, const Dtype* bottom_fidelity_sum, Dtype* normalized_bottom_fidelity_data,
			const Dtype* bottom_feature_data, Dtype* bottom_normalized_feature_data,
			const int N, const int C_f, const int G, const int C, const bool pre_scaled );
	static void reordered_whittened_input(
			const Dtype* X, const Dtype* X_bar, const Dtype* Phi, Dtype* Z,
			const int N, const int C, const int C_f, const int G, const int R, const bool mean_removal);
	static void forward_backscaling( const Dtype* YsupR, const Dtype* Omega, Dtype* Y,
			const int N, const int C_out, const int G_out, const int C_f, const int c_f );
	// Backward
	static void backward_scaling(
			const Dtype* y_diff, const Dtype* Omega, Dtype* YsupR_diff,
			const int N, const int C_out, const int G_out, const int C_f, const int c_f );
	static void filter_diff(
			const Dtype* W_tilde_diff, const Dtype* dh_dw_bar_bar, Dtype* W_diff,
			const int C, const int C_f, const int R, const int C_out, const int K, const int N,
			const bool isdeconv, const bool mean_adding_back );
	static void input_diff_scaling(
			const Dtype* Z_diff, const Dtype* Phi, Dtype* X_diff,
			const int N, const int C, const int C_f, const int G, const int R );
	static void input_diff_bar(
			const Dtype* dh_dX_bar, const Dtype* normalized_bottom_fidelity_data, Dtype* X_diff,
			const int N, const int C, const int C_f, const int G);
	static void input_fidelity_diff_scaling(
			const Dtype* Z_diff, const Dtype* X, Dtype* Phi_diff,
			const int N, const int C_f, const int G, const int R );
};


#ifndef CPU_ONLY

template<typename Dtype>
struct CommonFidaConvLayerAux<Dtype,mGPU> {
	// Forward
	static void reorder_filters(const Dtype* W, Dtype* W_tilde,
			const int C, const int C_f, const int R, const int C_out, const int K,
			const bool isdeconv );
	static void input_per_channel_mean(
			const Dtype* bottom_fidelity_data, const Dtype* bottom_fidelity_sum, Dtype* normalized_bottom_fidelity_data,
			const Dtype* bottom_feature_data, Dtype* bottom_normalized_feature_data,
			const int N, const int C_f, const int G, const int C, const bool pre_scaled );
	static void reordered_whittened_input(
			const Dtype* X, const Dtype* X_bar, const Dtype* Phi, Dtype* Z,
			const int N, const int C, const int C_f, const int G, const int R, const bool mean_removal);
	static void forward_backscaling( const Dtype* YsupR, const Dtype* Omega, Dtype* Y,
			const int N, const int C_out, const int G_out, const int C_f, const int c_f );
	// Backward
	static void backward_scaling(
			const Dtype* y_diff, const Dtype* Omega, Dtype* YsupR_diff,
			const int N, const int C_out, const int G_out, const int C_f, const int c_f );
	static void filter_diff(
			const Dtype* W_tilde_diff, const Dtype* dh_dw_bar_bar, Dtype* W_diff,
			const int C, const int C_f, const int R, const int C_out, const int K, const int N,
			const bool isdeconv, const bool mean_adding_back );
	static void input_diff_scaling(
			const Dtype* Z_diff, const Dtype* Phi, Dtype* X_diff,
			const int N, const int C, const int C_f, const int G, const int R );
	static void input_diff_bar(
			const Dtype* dh_dX_bar, const Dtype* normalized_bottom_fidelity_data, Dtype* X_diff,
			const int N, const int C, const int C_f, const int G);
	static void input_fidelity_diff_scaling(
			const Dtype* Z_diff, const Dtype* X, Dtype* Phi_diff,
			const int N, const int C_f, const int G, const int R );
};


#endif

}

#endif

