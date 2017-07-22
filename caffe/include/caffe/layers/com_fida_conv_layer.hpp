#ifndef CAFFE_COMMON_FIDA_CONV_HPP_
#define CAFFE_COMMON_FIDA_CONV_HPP_


#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "boost/shared_ptr.hpp"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/common_convx.hpp"
//#include "caffe/layers/pooling_layer.hpp"
//#include "caffe/layers/depooling_layer.hpp"
#include "caffe/layers/nd_conv_layer.hpp"

namespace caffe {


template <typename Dtype> class PoolingLayer;
template <typename Dtype> class DepoolingLayer;

template <typename Dtype>
class CommonConvolutionLayer<Dtype,ndConvLayer_T> : public ndConvolutionLayer<Dtype> {
public:
  typedef ndConvolutionLayer<Dtype> layer_t;
  typedef PoolingLayer<Dtype> pooling_t;
  explicit CommonConvolutionLayer(const LayerParameter& param)
	  : ndConvolutionLayer<Dtype>(param) {}
  static const char* conv_type() { return "ndConvolution"; }
  static const char* pool_type() { return "Pooling"; }
};

template <typename Dtype>
class CommonConvolutionLayer<Dtype,ndDeconvLayer_T> : public ndDeconvolutionLayer<Dtype> {
public:
  typedef ndDeconvolutionLayer<Dtype> layer_t;
  typedef DepoolingLayer<Dtype> pooling_t;
  explicit CommonConvolutionLayer(const LayerParameter& param)
	  : ndDeconvolutionLayer<Dtype>(param) {}
  static const char* conv_type() { return "ndDeconvolution"; }
  static const char* pool_type() { return "Depooling"; }
};

template <typename Dtype, int ConvOrDeconv_T>
class CommonFidaConvLayer : public CommonConvolutionLayer<Dtype,ConvOrDeconv_T> {
public:
	typedef CommonConvolutionLayer<Dtype,ConvOrDeconv_T>  conv_trait;
	typedef CommonConvolutionLayerX<Dtype,ConvOrDeconv_T> conv_layerx_t;
	typedef typename conv_trait::layer_t base_layer_t;
	typedef typename conv_trait::pooling_t base_pool_t;
	typedef CommonFidaConvLayer<Dtype,ConvOrDeconv_T> this_t;
	explicit CommonFidaConvLayer(const LayerParameter& param, bool parent_channel_as_geo = false)
	  : CommonConvolutionLayer<Dtype,ConvOrDeconv_T>(param),
	    parent_channel_as_geo_(parent_channel_as_geo) {}

	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);

	virtual inline int MinBottomBlobs() const { return 1; }
	virtual inline int MaxBottomBlobs() const { return 2; }
	virtual inline int ExactTopBlobs() const { return 1; }
    virtual inline bool EqualNumBottomTopBlobs() const { return false; }

protected:
    template<class DeviceMode>
	void Forward_generic(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);
    template<class DeviceMode>
	void Backward_generic(const vector<Blob<Dtype>*>& top,
	  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
	  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
	  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

protected:
	void set_up_bottom_top (const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

	void pre_forward(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

	void fidelity_conv_setup();
	void fidelity_conv_forward();
	void fidelity_conv_backward();

protected:

	shared_ptr<this_t> internal_fida_layer_;
    CommonInternalBottomTop<Dtype> fida_ibt;

    vector<int> bottom_feature_shape_;
    Blob<Dtype> sum_mult_;
    DynamicBlobAt<Dtype,SyncedMemoryAllocator::MIDDLE> buffer1_;
    //Blob<Dtype> buffer1_;
    Blob<Dtype> normalized_bottom_fidelity_;
    DynamicBlobAt<Dtype,SyncedMemoryAllocator::LARGE> bottom_buffer1_;
    //Blob<Dtype> bottom_buffer1_;

    /// mean
    Blob<Dtype> bottom_feature_mean_;

	/// fidelity column
    int fidelity_backend_;
    int fidelity_backend_actual_;
	shared_ptr< conv_layerx_t > fidelity_conv_layer_;
	shared_ptr< base_pool_t >  fidelity_pool_layer_;
	vector<bool> fidelity_propagate_down_vec_;

	Blob<Dtype> fidelity_conv_input_;   // shared with bottom[1] (bottom_fidelity_, Phi, reshaped to {N*C_f*1*G})
	Blob<Dtype> fidelity_conv_output_;  // an independent blob (Omega, reshaped to {N*C_f*1*G_out})
	Blob<Dtype> fidelity_backscaling_;	// shared with fidelity_conv_output_ (Omega, {N*C_f*G_out})
	vector<Blob<Dtype>*> fidelity_conv_bottom_, fidelity_conv_top_;	// blob vector for fidelity conv column
	int fidelity_channels_;	// C_f
	int fidelity_factor_;    // R
	shared_ptr< Blob<Dtype> > fixed_fidelity_conv_input_;	// use all one input when needed
	int bottom_geo_count_;	// G
	int top_geo_count_;		// G_out

	/// per-channel conv column
	Blob<Dtype> reordered_whittened_feature_; // an independent blob (Z)
	Blob<Dtype> reordered_weights_;         // an independent blob (W_tilde)
	int patch_count_;

	shared_ptr< conv_layerx_t > ch_conv_layer_;
	Blob<Dtype> ch_conv_input_;   // a sub blob (Z^{c_f})
	DynamicBlobAt<Dtype,SyncedMemoryAllocator::LARGE> ch_conv_output_;  // an independent blob (Y^R)
	// Blob<Dtype> ch_conv_output_;
	vector<Blob<Dtype>*> ch_conv_bottom_, ch_conv_top_;	// blob vector for per-channel conv column
	Blob<Dtype>* ch_conv_weights_;	// a sub blob (W^{c_f})
	vector<bool> ch_conv_propagate_down_vec_;

	/// sum of filter weights
	Blob<Dtype> weights_sum_;		// w_bar_bar

	/// pointers for the inputs and outputs (update in every function call)
	Blob<Dtype>* bottom_feature_;	// X
	Blob<Dtype>* bottom_fidelity_;	// Phi {N*C_f*G}
	Blob<Dtype>* top_feature_;		// Y
	vector<Blob<Dtype>*> my_bottom_, my_top_;

	/// pointers for filters
	Blob<Dtype>* weights_feature_;	// W
	Blob<Dtype>* bias_feature_;		// b

	/// fidelity implementation indicator
	bool compute_fidelity_conv_once_;
	bool ever_computed_fidelity_conv_;
	bool fidelity_fresh_reshaped_;
    
    /// mean removal indicator
    bool mean_removal_;
    bool mean_adding_back_;

    /// scaling up parameter
    bool back_scaling_up_;
    bool pre_scaled_;

    ///
    bool allow_pooling_backend_;

    ///
    bool channel_as_geo_;
    bool parent_channel_as_geo_;

    void set_parent_channel_as_geo( bool parent_channel_as_geo ) {
    	parent_channel_as_geo_ = parent_channel_as_geo;
    }

};

template <typename Dtype>
class FidaConvLayer : public CommonFidaConvLayer<Dtype, ndConvLayer_T> {
public:
	explicit FidaConvLayer(const LayerParameter& param)
	  : CommonFidaConvLayer<Dtype, ndConvLayer_T>(param) {}
	virtual inline const char*  type() const { return "FidaConv"; }
};

template <typename Dtype>
class FidaDeconvLayer : public CommonFidaConvLayer<Dtype, ndDeconvLayer_T> {
public:
	explicit FidaDeconvLayer(const LayerParameter& param)
	  : CommonFidaConvLayer<Dtype, ndDeconvLayer_T>(param) {}
	virtual inline const char*  type() const { return "FidaDeconv"; }
};

}

#endif

