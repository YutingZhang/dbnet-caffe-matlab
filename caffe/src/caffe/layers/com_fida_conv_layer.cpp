#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include <limits>
#include <vector>
#include <iostream>
#include <fstream>

#include "caffe/layers/com_fida_conv_layer.hpp"
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/layers/depooling_layer.hpp"
#include "caffe/layers_aux/fida_conv.hpp"


//#include "caffe/util/io.hpp"

using namespace std;

namespace caffe {

template <typename Dtype>
void CommonInternalBottomTop<Dtype>::bottom_top_channel_as_geo(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top,
		int update_internal_top ) {
	data_channel_as_geo( bottom, bottom_cag_, bottom_cag0_ , 0);
    for ( int i=0; i<bottom.size(); ++i )
        bottom_cag0_[i]->ShareDataDiff( *(bottom[i]) );
    if (update_internal_top>=TOP_FULL)
    	data_channel_as_geo( top, top_cag_, top_cag0_ , 1 );
    for ( int i=0; i<top.size(); ++i ) {
        auto s = top_cag_[i]->shape();
        if ( update_internal_top>=TOP_SHAPE && s.size() ) {
        	CHECK( s[channel_axis_+1]==1 ) << "Input channel should be set to 1 in output";
            s.erase(s.begin()+channel_axis_+1);
            top[i]->Reshape(s);
            top_cag0_[i]->ShareDataDiff( *(top[i]) );
        }
    }
}

template <typename Dtype>
void CommonInternalBottomTop<Dtype>::data_channel_as_geo(
		const vector<Blob<Dtype>*>& input, vector<Blob<Dtype>*>& output,
		vector<shared_ptr< Blob<Dtype> > >& output0, int channel_offset ) {
	if ( output.size()!=input.size() ) {
		output.clear();
		output.resize(input.size());
	}
	if ( output0.size()!=input.size() ) {
		output0.clear();
		output0.resize(input.size());
	}
	for ( int i=0; i<input.size(); ++i ) {
		vector<int> s = input[i]->shape();
        if (s.size()) {
		    s.insert( s.begin()+channel_axis_+channel_offset, 1 );
            if (output0[i]) {
		        output0[i]->Reshape( s );
            } else {
                output0[i].reset( new Blob<Dtype>(s) );
            }
        } else {
		    output0[i].reset(new Blob<Dtype>()); //->Reshape( vector<int>() );
        }
		output[i] = output0[i].get();
	}
}


template <typename Dtype>
void CommonInternalBottomTop<Dtype>::bottom_top_geo_as_channel(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top,
		int update_internal_top ) {
	data_geo_as_channel( bottom, bottom_cag_, bottom_cag0_, 0 );
    for ( int i=0; i<bottom.size(); ++i )
        bottom_cag0_[i]->ShareDataDiff( *(bottom[i]) );
    if (update_internal_top==TOP_FULL)
    	data_geo_as_channel( top, top_cag_, top_cag0_, 1 );
    for ( int i=0; i<top.size(); ++i ) {
        auto s = top_cag_[i]->shape();
        if ( update_internal_top>=TOP_SHAPE && s.size() ) {
            s.insert(s.begin()+channel_axis_+1,1);
            top[i]->Reshape(s);
            top_cag0_[i]->ShareDataDiff( *(top[i]) );
        }
    }
}

template <typename Dtype>
void CommonInternalBottomTop<Dtype>::data_geo_as_channel(
		const vector<Blob<Dtype>*>& input, vector<Blob<Dtype>*>& output,
		vector<shared_ptr< Blob<Dtype> > >& output0,int channel_offset ) {
	if ( output.size()!=input.size() ) {
		output.clear();
		output.resize(input.size());
	}
	if ( output0.size()!=input.size() ) {
		output0.clear();
		output0.resize(input.size());
	}
	for ( int i=0; i<input.size(); ++i ) {
		vector<int> s = input[i]->shape();
        if (s.size()) {
        	CHECK( s[channel_axis_+channel_offset]==1 ) << "Cannot remove channel with value >1";
		    s.erase( s.begin()+channel_axis_ );
            if (output0[i]) {
		        output0[i]->Reshape( s );
            } else {
                output0[i].reset( new Blob<Dtype>(s) );
            }
        } else {
		    output0[i].reset(new Blob<Dtype>()); //->Reshape( vector<int>() );
        }
		output[i] = output0[i].get();
	}
}


template <typename Dtype, int ConvOrDeconv_T>
void CommonConvolutionLayerX<Dtype, ConvOrDeconv_T>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top, bool fake_channel ) {
    fake_channel_ = fake_channel;

    this->common_conv_t::LayerSetUp( bottom, top );

    if ( !fake_channel_ ) return;

	conv_ibt.channel_axis_ = this->channel_axis_;	///IMPORTANT

	auto layer_param = this->layer_param_;

	layer_param.set_name( layer_param.name() + string(" [internal conv]") );

	ConvolutionParameter& conv_param = *(layer_param.mutable_convolution_param());

	{
		const int* pad_data    = this->pad_.cpu_data();
		int pad_dim = this->pad_.count();
		conv_param.clear_pad();
		CHECK( pad_data[0]==0 ) << "pad should be [0,...]";
		for ( int i=1; i<pad_dim; ++i ) {
			conv_param.add_pad( pad_data[i] );
		}
	}
	{
		const int* stride_data    = this->stride_.cpu_data();
		int stride_dim = this->stride_.count();
		conv_param.clear_stride();
		CHECK( stride_data[0]==1 ) << "stride should be [1,...]";
		for ( int i=1; i<stride_dim; ++i ) {
			conv_param.add_stride( stride_data[i] );
		}
	}
	{
		const int* kernel_data    = this->kernel_shape_.cpu_data();
		int kernel_dim = this->kernel_shape_.count();
		CHECK( kernel_data[0]>0 ) << "kernel_size should be [C,...]";
		conv_param.clear_kernel_size();
		for ( int i=1; i<kernel_dim; ++i ) {
			conv_param.add_kernel_size( kernel_data[i] );
		}
	}

	internal_conv_layer_.reset( new this_t(layer_param, false) );

	conv_ibt.bottom_top_geo_as_channel(bottom,top,CommonInternalBottomTop<Dtype>::TOP_FULL);
	internal_conv_layer_->SetUp( conv_ibt.bottom_cag_, conv_ibt.top_cag_ );
	conv_ibt.bottom_top_geo_as_channel(bottom,top,CommonInternalBottomTop<Dtype>::TOP_SHAPE);

	for ( int i=0; i<this->blobs_.size(); ++i ) {
		internal_conv_layer_->blobs_[i]->ShareDataDiff( *(this->blobs_[i]) );
	}

}
template <typename Dtype, int ConvOrDeconv_T>
void CommonConvolutionLayerX<Dtype, ConvOrDeconv_T>::Reshape(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
    if ( fake_channel_ ) {
        conv_ibt.bottom_top_geo_as_channel(bottom,top,CommonInternalBottomTop<Dtype>::TOP_FULL);
        internal_conv_layer_->Reshape( conv_ibt.bottom_cag_, conv_ibt.top_cag_ );
        conv_ibt.bottom_top_geo_as_channel(bottom,top,CommonInternalBottomTop<Dtype>::TOP_SHAPE);
    } else {
        this->common_conv_t::Reshape( bottom, top );
    }
}
template <typename Dtype, int ConvOrDeconv_T>
void CommonConvolutionLayerX<Dtype, ConvOrDeconv_T>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
    if ( fake_channel_ ) {
        conv_ibt.bottom_top_geo_as_channel(bottom,top);
        internal_conv_layer_->pub_Forward_cpu( conv_ibt.bottom_cag_, conv_ibt.top_cag_ );
    } else {
        this->common_conv_t::Forward_cpu( bottom, top );
    }
}
template <typename Dtype, int ConvOrDeconv_T>
void CommonConvolutionLayerX<Dtype, ConvOrDeconv_T>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
    if ( fake_channel_ ) {
        conv_ibt.bottom_top_geo_as_channel(bottom,top);
        internal_conv_layer_->pub_Forward_gpu( conv_ibt.bottom_cag_, conv_ibt.top_cag_ );
    } else {
        this->common_conv_t::Forward_gpu( bottom, top );
    }
}
template <typename Dtype, int ConvOrDeconv_T>
void CommonConvolutionLayerX<Dtype, ConvOrDeconv_T>::Backward_cpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if ( fake_channel_ ) {
        conv_ibt.bottom_top_geo_as_channel(bottom,top);
        internal_conv_layer_->pub_Backward_cpu( conv_ibt.top_cag_, propagate_down, conv_ibt.bottom_cag_ );
    } else {
        this->common_conv_t::Backward_cpu( top, propagate_down, bottom );
    }
}
template <typename Dtype, int ConvOrDeconv_T>
void CommonConvolutionLayerX<Dtype, ConvOrDeconv_T>::Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if ( fake_channel_ ) {
        conv_ibt.bottom_top_geo_as_channel(bottom,top);
        internal_conv_layer_->pub_Backward_gpu( conv_ibt.top_cag_, propagate_down, conv_ibt.bottom_cag_ );
    } else {
        this->common_conv_t::Backward_gpu( top, propagate_down, bottom );
    }
}

// -------------------------------------------------------------------------------

template <typename Dtype,int ConvOrDeconv_T>
void CommonFidaConvLayer<Dtype,ConvOrDeconv_T>::set_up_bottom_top (
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	bottom_feature_ = bottom[0];
	top_feature_    = top[0];

	if (fixed_fidelity_conv_input_.get())	// if fixed_fidelity is available then always use it
		bottom_fidelity_ = fixed_fidelity_conv_input_.get();
	else
		bottom_fidelity_ = bottom[1];

	if (my_bottom_.empty()) {
        my_bottom_.push_back( bottom_feature_ );
    } else {
        my_bottom_[0] = bottom_feature_;
    }
	if (my_top_.empty()) {
        my_top_.push_back( top_feature_ );
    } else {
        my_top_[0] = top_feature_;
    }

}


template <typename Dtype,int ConvOrDeconv_T>
void CommonFidaConvLayer<Dtype,ConvOrDeconv_T>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	CHECK_EQ( this->layer_param().convolution_param().group(), 1 ) <<
			"It only supports group == 1";

    {
    	const FidaConvParameter& fida_param = this->layer_param_.fida_conv_param();
		mean_removal_     = fida_param.mean_removal();
		mean_adding_back_ = fida_param.mean_adding_back();
		back_scaling_up_  = fida_param.back_scaling_up();
		pre_scaled_       = fida_param.pre_scaled();
		fidelity_backend_ = fida_param.fidelity_backend();
		channel_as_geo_   = fida_param.channel_as_geo();
    }

	ch_conv_propagate_down_vec_.clear();  ch_conv_propagate_down_vec_.push_back(true);
	fidelity_propagate_down_vec_.clear(); fidelity_propagate_down_vec_.push_back(true);

	set_up_bottom_top(bottom,top);

	base_layer_t::LayerSetUp( my_bottom_, my_top_ );
	CHECK_EQ( this->parallel_batch_size_, 1) <<
			"parallel_batch_size_ must be 1";

	// guarantee Reshape can be triggered -------------------------------------
    this->num_=numeric_limits<decltype(this->num_)>::max();
    fidelity_channels_ = numeric_limits<decltype(fidelity_channels_)>::max();
    // ------------------------------------------------------------------------

    compute_fidelity_conv_once_  = false;
    ever_computed_fidelity_conv_ = false;
    fidelity_fresh_reshaped_ = false;

    if (channel_as_geo_) {
        fida_ibt.channel_axis_ = this->channel_axis_;
    	auto layer_param = this->layer_param_;
    	layer_param.set_name( layer_param.name() + string(" [internal fida]") );
    	FidaConvParameter&    fida_param = *(layer_param.mutable_fida_conv_param());
    	ConvolutionParameter& conv_param = *(layer_param.mutable_convolution_param());
    	fida_param.set_channel_as_geo(false);

        {
            const int* pad_data    = this->pad_.cpu_data();
            int pad_dim = this->pad_.count();
            conv_param.clear_pad();
            conv_param.add_pad( 0 );
			for ( int i=0; i<pad_dim; ++i )
				conv_param.add_pad( pad_data[i] );
        }
        {
            const int* stride_data    = this->stride_.cpu_data();
            int stride_dim = this->stride_.count();
            conv_param.clear_stride();
            conv_param.add_stride( 1 );
			for ( int i=0; i<stride_dim; ++i )
				conv_param.add_stride( stride_data[i] );
        }
        {
            const int* kernel_data    = this->kernel_shape_.cpu_data();
            int kernel_dim = this->kernel_shape_.count();
            conv_param.clear_kernel_size();
            conv_param.add_kernel_size( this->channels_ );
			for ( int i=0; i<kernel_dim; ++i )
				conv_param.add_kernel_size( kernel_data[i] );
        }

    	internal_fida_layer_.reset( new this_t(layer_param, true) );

        {
    	    fida_ibt.bottom_top_channel_as_geo( bottom, top, CommonInternalBottomTop<Dtype>::TOP_FULL );
    	    internal_fida_layer_->SetUp( fida_ibt.bottom_cag_, fida_ibt.top_cag_ );
    	    fida_ibt.bottom_top_channel_as_geo( bottom, top, CommonInternalBottomTop<Dtype>::TOP_SHAPE );
        }

    	for ( int i=0; i<this->blobs_.size(); ++i ) {
    		internal_fida_layer_->blobs_[i]->ShareDataDiff( *(this->blobs_[i]) );
        }

    }

}

template <typename Dtype,int ConvOrDeconv_T>
void CommonFidaConvLayer<Dtype,ConvOrDeconv_T>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	if ( internal_fida_layer_.get() ) {
    	fida_ibt.bottom_top_channel_as_geo( bottom, top, CommonInternalBottomTop<Dtype>::TOP_FULL );
		internal_fida_layer_->Reshape( fida_ibt.bottom_cag_, fida_ibt.top_cag_ );
    	fida_ibt.bottom_top_channel_as_geo( bottom, top, CommonInternalBottomTop<Dtype>::TOP_SHAPE );
		return;
	}

	if ( bottom.size()==1 ) {
        bool b = fixed_fidelity_conv_input_.get() && 
            fixed_fidelity_conv_input_->num_axes()==bottom[0]->num_axes();
        for ( int i=0; b && i<fixed_fidelity_conv_input_->num_axes(); ++i ) {
            if ( i == this->channel_axis_ ) continue;
            b &= (fixed_fidelity_conv_input_->shape(i)==bottom[0]->shape(i));
        }
		if ( !b ) {
			// generate default fidelity map
			vector<int> default_fidelity_conv_input_shape = bottom[0]->shape();
			default_fidelity_conv_input_shape[this->channel_axis_] = 1;
			fixed_fidelity_conv_input_.reset( new Blob<Dtype>(
					default_fidelity_conv_input_shape) );
			Dtype*  default_fidelity_conv_input_data =
					fixed_fidelity_conv_input_->mutable_cpu_data();
			caffe_set( fixed_fidelity_conv_input_->count(), Dtype(1.), default_fidelity_conv_input_data );
			compute_fidelity_conv_once_ = true;
			pre_scaled_ = true;
		}
	} else {
        bool b = bottom[0]->num_axes() == bottom[1]->num_axes();
        for ( int i=0; b && i<bottom[0]->num_axes() ; ++i ) {
            if ( i == this->channel_axis_ ) continue;
            b &= (bottom[0]->shape(i)==bottom[1]->shape(i));
        }
		CHECK( b ) << "the two bottom blobs should have the same shape (except for channels)";
		compute_fidelity_conv_once_ = this->layer_param_.fida_conv_param().compute_fidelity_once();
	}

	set_up_bottom_top(bottom,top);

    bool reshape_fidelity_conv = false, reshape_ch_conv = false;
    {
        bool b = (bottom_feature_->shape().size()==bottom_feature_shape_.size());
        for ( int i=0; b && i<bottom_feature_shape_.size(); ++i ) {
            if ( i == this->channel_axis_ ) continue;
            b &= (bottom_feature_->shape(i)==bottom_feature_shape_[i]);
        }
        if (!b)
            reshape_fidelity_conv = reshape_ch_conv = true;
        else {
            if (this->channels_!=my_bottom_[0]->shape(this->channel_axis_)) reshape_ch_conv = true;
            // something to be solved here
            if (fidelity_channels_!=bottom_fidelity_->shape(this->channel_axis_)) {
                reshape_fidelity_conv = true;
                reshape_ch_conv = true;
            }
        }
    }

	fidelity_channels_ = bottom_fidelity_->shape( this->channel_axis_ );
	CHECK( !(this->channels_%fidelity_channels_) ) << "channels_ should be divisible by fidelity_channels_";
	fidelity_factor_ = this->channels_ / fidelity_channels_;

	// set up the internal conv layer
	if (reshape_ch_conv) {

		// set up the original layer
	    base_layer_t::Reshape( my_bottom_, my_top_ );

	    if ( parent_channel_as_geo_ ) {
	    	auto s = top_feature_->shape();
	    	s[this->channel_axis_+1] = 1;
	    	top_feature_->Reshape(s);
	    }

        bottom_feature_shape_= bottom_feature_->shape();

	    // pointer for original parameters
	    weights_feature_ = this->blobs_[0].get();
	    if ( this->bias_term_ )
	    	bias_feature_ = this->blobs_[1].get();
	    else
	    	bias_feature_ = NULL;

	    // blob for reordered whittened features
	    vector<int> ch_conv_input_shape, reordered_feature_shape;
	    {
	    	reordered_feature_shape = bottom_feature_->shape();
	    	reordered_feature_shape[this->channel_axis_] = fidelity_factor_;
	    	reordered_feature_shape.insert( reordered_feature_shape.begin(), fidelity_channels_ );
	    	ch_conv_input_shape = vector<int>( reordered_feature_shape.begin()+1, reordered_feature_shape.end() );
	    }
	    reordered_whittened_feature_.Reshape( reordered_feature_shape );

	    // blob for reordered weights
	    vector<int> reordered_weights_shape;
	    {
	    	reordered_weights_shape = weights_feature_->shape();
            if (this->reverse_dimensions())
	    	    reordered_weights_shape[0] = fidelity_factor_;
            else
                reordered_weights_shape[1] = fidelity_factor_;
	    	reordered_weights_shape.insert( reordered_weights_shape.begin(), fidelity_channels_ );
	    }
	    reordered_weights_.Reshape(reordered_weights_shape);
	    patch_count_ = weights_feature_->count(2);

	    // blob for weights sum
        //if ( mean_removal_ ) {
            if (this->reverse_dimensions())
                weights_sum_.Reshape( vector<int>({this->channels_,this->num_output_}) );
            else
                weights_sum_.Reshape( vector<int>({this->num_output_,this->channels_}) );
        //}
	    // set up per-channel convolution layer
		LayerParameter param;
        param.set_name( this->layer_param().name() + " [fida-channel-conv]" );
        param.set_type( conv_trait::conv_type() );
		ConvolutionParameter* conv_param = param.mutable_convolution_param();
        *conv_param = this->layer_param().convolution_param();
		conv_param->set_bias_term( false );
		conv_param->set_single_channel_parallel( true );
		// conv_param->set_parallel_batch_size( this->num_ );
		ch_conv_layer_.reset( new conv_layerx_t(param, parent_channel_as_geo_) );

		ch_conv_input_.Reshape( ch_conv_input_shape );
	    ch_conv_bottom_.clear(); ch_conv_top_.clear();
		ch_conv_bottom_.push_back( &ch_conv_input_ );
		ch_conv_top_.push_back( &ch_conv_output_ );
		ch_conv_layer_->SetUp(ch_conv_bottom_, ch_conv_top_);

		ch_conv_weights_ = ch_conv_layer_->blobs()[0].get();

    	// init count
		bottom_geo_count_ = bottom_feature_->count(this->channel_axis_+1);
	    top_geo_count_    = top_feature_->count(this->channel_axis_+1);

	    // mean blob
        //if ( mean_removal_ ) {
            const vector<int> bottom_feature_shape = bottom_feature_->shape();
            bottom_feature_mean_.Reshape( vector<int>(bottom_feature_shape.begin(),
                    bottom_feature_shape.begin()+this->channel_axis_+1) );
        //}
	    // prod buffer
	    buffer1_.Reshape( vector<int>({std::max({
	    	this->num_*this->num_output_,
	    	this->num_*this->channels_,
	    	this->num_*this->num_output_+
	    	std::max({this->channels_*this->num_output_,this->num_*this->channels_}) })}) );
	    // sum mult
	    sum_mult_.Reshape( vector<int>({std::max( {bottom_geo_count_, this->num_,
	    	patch_count_, top_geo_count_, this->num_output_, this->channels_} )} ) );
	    {
	    	Dtype* sum_mult_data = sum_mult_.mutable_cpu_data();
	    	caffe_set( sum_mult_.count(), Dtype(1.), sum_mult_data );
	    }

	    //
	    bottom_buffer1_.ReshapeLike( *bottom_feature_ );

	}

	// set up re-weighting conv layer
	if (reshape_fidelity_conv) {

		allow_pooling_backend_ = true;
		for ( int t=0; t<2; t++ ) {
			fidelity_conv_setup();

			vector<int> fidelity_backscaling_shape = top_feature_->shape();
			fidelity_backscaling_shape[this->channel_axis_] = bottom_fidelity_->shape(this->channel_axis_);

			if ( !t && fidelity_conv_output_.shape() != fidelity_backscaling_shape ) {
				allow_pooling_backend_ = false;
				continue;
			}

			fidelity_backscaling_.Reshape(fidelity_backscaling_shape);
			fidelity_backscaling_.ShareDataDiff( fidelity_conv_output_ );	// this is permanent share

			// normalized fidelity for mean computation
			normalized_bottom_fidelity_.ReshapeLike( *bottom_fidelity_ );

			// init count
			ever_computed_fidelity_conv_ = false;
			fidelity_fresh_reshaped_ = true;

			break;
		}
		allow_pooling_backend_ = allow_pooling_backend_ && !parent_channel_as_geo_;
	}

}

template <typename Dtype,int ConvOrDeconv_T>
void CommonFidaConvLayer<Dtype,ConvOrDeconv_T>::fidelity_conv_setup() {

	if ( allow_pooling_backend_ ) {
		fidelity_backend_actual_ = fidelity_backend_;
		if ( fidelity_backend_ == FidaConvParameter_FidelityBackend_AUTO ) {
			vector<int> fidelity_orig_input_shape = bottom_fidelity_->shape();
			if ( fidelity_orig_input_shape.size()-3 == this->channel_axis_ ) {
				// if 2-D geo (Pooling is only implemented for 2D)
				fidelity_backend_actual_ = FidaConvParameter_FidelityBackend_POOL;
			} else {
				// if other-D geo
				fidelity_backend_actual_ = FidaConvParameter_FidelityBackend_CONV;
			}
		}
	} else {
		fidelity_backend_actual_ = FidaConvParameter_FidelityBackend_CONV;
	}

	Layer<Dtype>* fidelity_op_layer;
    LayerParameter param;

	switch ( fidelity_backend_actual_ ) {
	case FidaConvParameter_FidelityBackend_CONV:
	{
        param.set_name( this->layer_param().name() + " [fida-fidelity-conv]" );
        param.set_type( conv_trait::conv_type() );
		ConvolutionParameter* conv_param = param.mutable_convolution_param();
		*conv_param = this->layer_param().convolution_param();
		conv_param->set_single_channel_parallel( true );
		FillerParameter* weight_filler = conv_param->mutable_weight_filler();
		weight_filler->set_type( "constant" );
		weight_filler->set_value( Dtype(1.) );
		conv_param->set_num_output( 1 );
		conv_param->set_bias_term( false );
		conv_param->set_axis( this->channel_axis_+1 );
		fidelity_conv_layer_.reset( new conv_layerx_t(param, parent_channel_as_geo_) );
        fidelity_op_layer = fidelity_conv_layer_.get();

		vector<int> fidelity_conv_input_shape = bottom_fidelity_->shape();
		fidelity_conv_input_shape.insert(fidelity_conv_input_shape.begin()+
				this->channel_axis_+1, 1 );
		fidelity_conv_input_.Reshape( fidelity_conv_input_shape );

		break;
	}

	case FidaConvParameter_FidelityBackend_POOL:
	{
        param.set_name( this->layer_param().name() + " [fida-fidelity-pool]" );
        param.set_type( conv_trait::pool_type() );
		vector<int> fidelity_orig_input_shape = bottom_fidelity_->shape();
		size_t fidelity_ndim = fidelity_orig_input_shape.size();
		CHECK( fidelity_orig_input_shape.size()-3 == this->channel_axis_ ) <<
				"Only support 2-D geo input when using Pooling and the FidelityBackend";
		PoolingParameter* pooling_param = param.mutable_pooling_param();
		pooling_param->set_pool( PoolingParameter_PoolMethod_SUM );

        {
            const int* pad_data    = this->pad_.cpu_data();
            int pad_dim = this->pad_.count();
            CHECK(pad_dim==2) << "only support 2-d image";
			pooling_param->set_pad_h( pad_data[0] );
			pooling_param->set_pad_w( pad_data[1] );
        }
        {
            const int* stride_data    = this->stride_.cpu_data();
            int stride_dim = this->stride_.count();
            CHECK(stride_dim==2) << "only support 2-d image";
			pooling_param->set_stride_h( stride_data[0] );
			pooling_param->set_stride_w( stride_data[1] );
        }
        {
            const int* kernel_data    = this->kernel_shape_.cpu_data();
            int kernel_dim = this->kernel_shape_.count();
            CHECK(kernel_dim==2) << "only support 2-d image";
			pooling_param->set_kernel_h( kernel_data[0] );
			pooling_param->set_kernel_w( kernel_data[1] );
        }
		fidelity_pool_layer_.reset( new base_pool_t(param) );
        fidelity_op_layer = fidelity_pool_layer_.get();

		int  input_num = 1;
		for ( size_t i = 0; i<fidelity_ndim-3; ++i )
			input_num *= fidelity_orig_input_shape[i];
		vector<int> fidelity_conv_input_shape( {
			input_num,
			fidelity_orig_input_shape[fidelity_ndim-3],
			fidelity_orig_input_shape[fidelity_ndim-2],
			fidelity_orig_input_shape[fidelity_ndim-1] } );
		fidelity_conv_input_.Reshape( fidelity_conv_input_shape );

		break;
	}

	default:
		LOG(FATAL) << "Unrecognized fidelity_backend";
	}

	fidelity_conv_bottom_.clear(); fidelity_conv_top_.clear();
	fidelity_conv_bottom_.push_back( &fidelity_conv_input_ );
	fidelity_conv_top_.push_back( &fidelity_conv_output_ );
	fidelity_op_layer->SetUp(fidelity_conv_bottom_, fidelity_conv_top_);

}

template <typename Dtype,int ConvOrDeconv_T>
void CommonFidaConvLayer<Dtype,ConvOrDeconv_T>::fidelity_conv_forward() {
	fidelity_conv_input_.ShareDataDiff( *bottom_fidelity_ );
	switch ( fidelity_backend_actual_ ) {
	case FidaConvParameter_FidelityBackend_CONV:
		fidelity_conv_layer_->Forward( fidelity_conv_bottom_, fidelity_conv_top_ );
		break;
	case FidaConvParameter_FidelityBackend_POOL:
		fidelity_pool_layer_->Forward( fidelity_conv_bottom_, fidelity_conv_top_ );
		break;
	default:
		LOG(FATAL) << "Unrecognized fidelity_backend";
	}
}

template <typename Dtype,int ConvOrDeconv_T>
void CommonFidaConvLayer<Dtype,ConvOrDeconv_T>::fidelity_conv_backward() {
	fidelity_conv_input_.ShareDataDiff( *bottom_fidelity_ );
	switch ( fidelity_backend_actual_ ) {
	case FidaConvParameter_FidelityBackend_CONV:
		fidelity_conv_layer_->Backward( fidelity_conv_top_, fidelity_propagate_down_vec_, fidelity_conv_bottom_ );
		break;
	case FidaConvParameter_FidelityBackend_POOL:
		fidelity_pool_layer_->Backward( fidelity_conv_top_, fidelity_propagate_down_vec_, fidelity_conv_bottom_ );
		break;
	default:
		LOG(FATAL) << "Unrecognized fidelity_backend";
	}
}


template <typename Dtype,int ConvOrDeconv_T>
void CommonFidaConvLayer<Dtype,ConvOrDeconv_T>::pre_forward(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top ) {
	if ( bottom.size()>1 && fidelity_fresh_reshaped_ &&
			!ever_computed_fidelity_conv_ && compute_fidelity_conv_once_ ) {
		// test whether to shrink the bottom fidelity map
		const Dtype* q,* p,* p0;
        Dtype* buf;
		Blob<Dtype> buf_blob( vector<int>({1, bottom_geo_count_}) );
		bool channel_identical = true;
		if (Caffe::mode() == Caffe::GPU ) {
#ifndef CPU_ONLY
			p   = p0 = bottom[1]->gpu_data();
			buf = buf_blob.mutable_gpu_data();
			for ( int n = 0; n<this->num_ && channel_identical; ++n ) {
				q = p;
				for ( int c=0; c<fidelity_channels_ && channel_identical; ++c ) {
					Dtype l1dist;
					caffe_gpu_sub<Dtype>( bottom_geo_count_, p, q, buf);
					caffe_gpu_asum<Dtype>( bottom_geo_count_, buf, &l1dist );
					p += bottom_geo_count_;
					channel_identical = channel_identical && (l1dist < std::numeric_limits<Dtype>::epsilon());
				}
			}
#endif
		} else {
			p   = p0 = bottom[1]->cpu_data();
			buf = buf_blob.mutable_cpu_data();
			for ( int n = 0; n<this->num_ && channel_identical; ++n ) {
				q = p;
				for ( int c=0; c<fidelity_channels_ && channel_identical; ++c ) {
					Dtype l1dist;
					caffe_sub<Dtype>( bottom_geo_count_, p, q, buf);
					l1dist = caffe_cpu_asum<Dtype>( bottom_geo_count_, buf );
					p += bottom_geo_count_;
					channel_identical = channel_identical && (l1dist < std::numeric_limits<Dtype>::epsilon());
				}
			}
		}

		if ( channel_identical ) {
			vector<int> fixed_fidelity_conv_shape = bottom[1]->shape();
			fixed_fidelity_conv_shape[this->channel_axis_] = 1;
			fixed_fidelity_conv_input_.reset( new Blob<Dtype>( fixed_fidelity_conv_shape ) );
			Dtype* fixed_fidelity_conv_input_data;
			if (Caffe::mode() == Caffe::GPU ) {
#ifndef CPU_ONLY
				fixed_fidelity_conv_input_data = fixed_fidelity_conv_input_->mutable_gpu_data();
#endif
			} else {
				fixed_fidelity_conv_input_data = fixed_fidelity_conv_input_->mutable_cpu_data();;
			}
			zcaffe_blockcopy( this->num_, bottom_geo_count_,
					fidelity_channels_*bottom_geo_count_, p0, bottom_geo_count_, fixed_fidelity_conv_input_data );
			this->Reshape( bottom, top );
		}
	}
	fidelity_fresh_reshaped_ = false;
}


/* --------------------------------------------------------------------------------------------------------
 * --------------------------------------------------------------------------------------------------------
 */

template <typename Dtype,int ConvOrDeconv_T>
template <class DeviceMode>
void CommonFidaConvLayer<Dtype,ConvOrDeconv_T>::Forward_generic(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {

	if ( internal_fida_layer_.get() ) {
    	fida_ibt.bottom_top_channel_as_geo( bottom, top );
		internal_fida_layer_->template Forward_generic<DeviceMode>(fida_ibt.bottom_cag_, fida_ibt.top_cag_);
		return;
	}

	//static int iter = 0;

	pre_forward( bottom, top );

	set_up_bottom_top(bottom,top);
  
	SyncedMemoryGuard buffer1_guard = buffer1_.hold_data();

	// compute_fidelity_backscaling
	if ( !compute_fidelity_conv_once_ || !ever_computed_fidelity_conv_ ) {
		fidelity_conv_forward();
        Dtype* Omega = fidelity_conv_output_.mutable_data( DeviceMode() );
        Dtype back_scaling_factor = (back_scaling_up_)?(patch_count_):(1);
        gcm<DeviceMode>::safeinv( fidelity_conv_output_.count(), Omega, Omega, back_scaling_factor );
		ever_computed_fidelity_conv_ = true;
		//ArrayToGZ("./snapshot/fida/fidelity_conv_output-"+std::to_string(iter)+".blob.gz",
		//		fidelity_conv_output_.shape(), fidelity_conv_output_.data( DeviceMode() ) );
	}

	// reordering filters
	{
		const Dtype* W = weights_feature_->data( DeviceMode() );
		Dtype* W_tilde = reordered_weights_.mutable_data( DeviceMode() );
		int C = this->channels_, C_f = fidelity_channels_, R = fidelity_factor_,
				C_out = this->num_output_, K = patch_count_;
		CommonFidaConvLayerAux<Dtype,DeviceMode>::reorder_filters(W, W_tilde,
				C, C_f, R, C_out, K,this->reverse_dimensions());
		//ArrayToGZ("./snapshot/fida/reordered_weights-"+std::to_string(iter)+".blob.gz",
		//		reordered_weights_.shape(), reordered_weights_.data( DeviceMode() ) );
	}


	if ( mean_removal_ ) {
		// compute filter per-channel sum (w_bar_bar)
		if ( mean_adding_back_ ) {
			Dtype* weights_sum_data = weights_sum_.mutable_data( DeviceMode() );
			const Dtype* weights_feature_data = weights_feature_->data( DeviceMode() );
			const Dtype* sum_mult_data = sum_mult_.data( DeviceMode() );
			gcm<DeviceMode>::gemv( CblasNoTrans, this->num_output_*this->channels_, patch_count_,
					Dtype(1.), weights_feature_data, sum_mult_data, Dtype(0.), weights_sum_data );
			//ArrayToGZ("./snapshot/fida/weights_sum-"+std::to_string(iter)+".blob.gz",
			//		weights_sum_.shape(), weights_sum_.data( DeviceMode() ) );
		}

		// compute input feature per-channel mean
		{
			SyncedMemoryGuard bottom_buffer1_guard = bottom_buffer1_.hold_data();

			Dtype* normalized_bottom_fidelity_data = normalized_bottom_fidelity_.mutable_data( DeviceMode() );
			const Dtype* sum_mult_data = sum_mult_.mutable_data( DeviceMode() );
			const Dtype* bottom_fidelity_data = bottom_fidelity_->data( DeviceMode() );
			Dtype* bottom_fidelity_sum = buffer1_.mutable_data( DeviceMode() );
			const Dtype* bottom_feature_data  = bottom_feature_->data( DeviceMode() );
			Dtype* bottom_normalized_feature_data = bottom_buffer1_.mutable_data( DeviceMode() );
			Dtype* bottom_feature_mean_data   = bottom_feature_mean_.mutable_data( DeviceMode() );

			gcm<DeviceMode>::gemv(CblasNoTrans, this->num_*fidelity_channels_, bottom_geo_count_,
					Dtype(1.), bottom_fidelity_data, sum_mult_data,
					Dtype(0.), bottom_fidelity_sum );

			{
				int N = this->num_, C_f = fidelity_channels_, G = bottom_geo_count_,
                    C = this->channels_;
				CommonFidaConvLayerAux<Dtype, DeviceMode>::input_per_channel_mean(
						bottom_fidelity_data, bottom_fidelity_sum, normalized_bottom_fidelity_data,
						bottom_feature_data, bottom_normalized_feature_data,
						N, C_f, G, C, pre_scaled_ );
			}

			gcm<DeviceMode>::gemv(CblasNoTrans, this->num_*this->channels_, bottom_geo_count_,
					Dtype(1.), bottom_normalized_feature_data, sum_mult_data,
					Dtype(0.), bottom_feature_mean_data );
		}
	}

	// compute_reordered_whittened_input
	{
		const Dtype* X     = bottom_feature_->data( DeviceMode() );
		const Dtype* X_bar = bottom_feature_mean_.data( DeviceMode() );
		const Dtype* Phi   = (pre_scaled_)?(NULL):(bottom_fidelity_->data( DeviceMode() ));
		Dtype* Z = reordered_whittened_feature_.mutable_data( DeviceMode() );
		int N = this->num_, C = this->channels_, C_f = fidelity_channels_,
				G = bottom_geo_count_, R = fidelity_factor_;
		CommonFidaConvLayerAux<Dtype, DeviceMode>::reordered_whittened_input(
				X, X_bar, Phi, Z, N, C, C_f, G, R, mean_removal_);
		//ArrayToGZ("./snapshot/fida/reordered_whittened_feature-"+std::to_string(iter)+".blob.gz",
		//		reordered_whittened_feature_.shape(), reordered_whittened_feature_.data( DeviceMode() ) );
	}

    //LOG(INFO) << "Use buffer1";
	// fill_output_with_bias
	{
		Dtype* top_feature_data = top_feature_->mutable_data( DeviceMode() );
		const Dtype* sum_mult_data = sum_mult_.data( DeviceMode() );
		Dtype* buf = buffer1_.mutable_data( DeviceMode() );
		if ( this->bias_term_ ) {
			const Dtype* bias_feature_data = bias_feature_->data( DeviceMode() );
			gcm<DeviceMode>::gemm(CblasNoTrans,CblasNoTrans, this->num_, this->num_output_,
					1, Dtype(1.), sum_mult_data, bias_feature_data, Dtype(0.), buf );
		} else {
			gcm<DeviceMode>::set(this->num_*this->num_output_, Dtype(0.), buf );
		}
        if ( mean_removal_ && mean_adding_back_ ) {
    		const Dtype* bottom_feature_mean_data   = bottom_feature_mean_.data( DeviceMode() );
    		const Dtype* weights_sum_data  = weights_sum_.data( DeviceMode() );
            gcm<DeviceMode>::gemm(CblasNoTrans,((this->reverse_dimensions())?CblasNoTrans:CblasTrans),
                    this->num_, this->num_output_,
                    this->channels_, Dtype(1.), bottom_feature_mean_data, weights_sum_data,
                    Dtype(1.), buf );
        }
		gcm<DeviceMode>::gemm(CblasNoTrans,CblasNoTrans, this->num_*this->num_output_, top_geo_count_,
				1, Dtype(1.), buf, sum_mult_data, Dtype(0.),  top_feature_data );
		//ArrayToGZ("./snapshot/fida/top_feature_bias-"+std::to_string(iter)+".blob.gz",
		//		top_feature_->shape(), top_feature_->data( DeviceMode() ) );

	}
    //LOG(INFO) << "Use buffer1: DONE";

	// convolution_and_backscaling
	{
		SyncedMemoryGuard ch_conv_output_guard = ch_conv_output_.hold_data();
		const Dtype* Omega = fidelity_conv_output_.data( DeviceMode() );
		for (int c_f=0; c_f<fidelity_channels_; ++c_f) {
			ch_conv_input_.ShareSubBlob( reordered_whittened_feature_, c_f );
			ch_conv_weights_->ShareSubBlob( reordered_weights_, c_f );
			ch_conv_layer_->Forward( ch_conv_bottom_, ch_conv_top_ );

		    //ArrayToGZ("./snapshot/fida/ch_conv_output-cf"+std::to_string(c_f)+
            //        "-"+std::to_string(iter)+".blob.gz",
            //        top_feature_->shape(), top_feature_->data( DeviceMode() ) );
			const Dtype* YsupR = ch_conv_output_.data( DeviceMode() );
			int N = this->num_, C_out = this->num_output_,
					G_out = top_geo_count_, C_f = fidelity_channels_;
			{

				Dtype* Y = top_feature_->mutable_data( DeviceMode() );
				CommonFidaConvLayerAux<Dtype, DeviceMode>::forward_backscaling(
						YsupR, Omega, Y, N, C_out, G_out, C_f, c_f );
			}
		}
	}

	//++iter;

}

template <typename Dtype,int ConvOrDeconv_T>
template <class DeviceMode>
void CommonFidaConvLayer<Dtype,ConvOrDeconv_T>::Backward_generic(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	if ( internal_fida_layer_.get() ) {
    	fida_ibt.bottom_top_channel_as_geo( bottom, top );
		internal_fida_layer_->template Backward_generic<DeviceMode>(
                fida_ibt.top_cag_, propagate_down, fida_ibt.bottom_cag_);
		return;
	}


    //LOG(INFO) << "Fida: Backward";
	set_up_bottom_top(bottom,top);

	bool propagate_fidelity = !compute_fidelity_conv_once_ && propagate_down[1];
	if ( propagate_fidelity ) {	// input fidelity
		if ( mean_removal_ ) {
			LOG(WARNING) << "Backpropagation is not implemented for mean_removal case. DISABLED";
			propagate_fidelity = false;
		}
	}

	SyncedMemoryGuard buffer1_guard = buffer1_.hold_data();

    Dtype* buf_base = buffer1_.mutable_data( DeviceMode() );
    const Dtype* sample_bias_diff = buf_base;

	// bias
    bool back_propagate_bias = this->bias_term_ && this->param_propagate_down_[1];
	if ( back_propagate_bias || ( mean_adding_back_ &&
			(this->param_propagate_down_[0] || propagate_down[0]) ) ) {
	    int N = this->num_, C_out = this->num_output_, G_out = top_geo_count_;

    	Dtype* sample_bias_diff = buf_base;
    	buf_base += N*C_out;

    	const Dtype* sum_mult_data = sum_mult_.data( DeviceMode() );
		const Dtype* y_diff = top_feature_->diff( DeviceMode() );

        gcm<DeviceMode>::gemv( CblasNoTrans, N*C_out, G_out,
                Dtype(1.), y_diff, sum_mult_data, Dtype(0.), sample_bias_diff );
        if ( back_propagate_bias ) {
    	    Dtype* bias_diff = bias_feature_->mutable_diff( DeviceMode() );
			gcm<DeviceMode>::gemv( CblasTrans, N, C_out,
					Dtype(1.), sample_bias_diff, sum_mult_data, Dtype(1.), bias_diff);
        }

	}

	// per-channel convolution
	if ( this->param_propagate_down_[0] || propagate_down[0] || propagate_fidelity ) {
		SyncedMemoryGuard ch_conv_output_guard = ch_conv_output_.hold_diff();
		//gcm<DeviceMode>::set(reordered_whittened_feature_.count(), Dtype(0), reordered_whittened_feature_.mutable_diff( DeviceMode() ));
		gcm<DeviceMode>::set(reordered_weights_.count(), Dtype(0), reordered_weights_.mutable_diff( DeviceMode() ));

		Dtype* Omega_diff = NULL; 
        Dtype back_scaling_up_factor = (back_scaling_up_)?Dtype(1.)/Dtype(patch_count_):Dtype(1.);
	    SyncedMemoryGuard ch_conv_output_guard2;
		if ( propagate_fidelity ) {
			ch_conv_output_guard2 = ch_conv_output_.hold_data();
			Omega_diff = fidelity_conv_output_.mutable_diff( DeviceMode() );
			// gcm<DeviceMode>::set(fidelity_conv_output_.count(), Dtype(0), Omega_diff);
		}
		{
			int N = this->num_, C_out = this->num_output_, C_f = fidelity_channels_, G_out = top_geo_count_;
			Dtype* YsupR_diff = ch_conv_output_.mutable_diff( DeviceMode() );
			const Dtype* y_diff = top_feature_->diff( DeviceMode() );
			const Dtype* Omega = fidelity_conv_output_.data( DeviceMode() );

			for (int c_f=0; c_f<fidelity_channels_; ++c_f) {
				CommonFidaConvLayerAux<Dtype, DeviceMode>::backward_scaling(
						y_diff, Omega, YsupR_diff, N, C_out, G_out, C_f, c_f );
				ch_conv_input_.ShareSubBlob( reordered_whittened_feature_, c_f );
				ch_conv_weights_->ShareSubBlob( reordered_weights_, c_f );
				ch_conv_layer_->Backward( ch_conv_top_, ch_conv_propagate_down_vec_, ch_conv_bottom_ );
				if ( propagate_fidelity ) {
					ch_conv_layer_->Forward( ch_conv_bottom_, ch_conv_top_ );
					Dtype* YsupR = ch_conv_output_.mutable_data(  DeviceMode() );
					Dtype* Omega_ch_diff_e = YsupR;	// reuse buffer

					const Dtype* sum_mult_data = sum_mult_.data( DeviceMode() );

					gcm<DeviceMode>::mul( ch_conv_output_.count(), y_diff, YsupR, Omega_ch_diff_e );
					for ( int n=0; n<N; ++n ) {
						Dtype* Omega_ch_diff_e_n = Omega_ch_diff_e + n*C_out*G_out;
						Dtype* Omega_diff_n_cf = Omega_diff + (n*C_f+c_f)*G_out;
						gcm<DeviceMode>::gemv( CblasTrans, C_out, G_out,
								Dtype(-back_scaling_up_factor), Omega_ch_diff_e_n, sum_mult_data,
								Dtype(0.), Omega_diff_n_cf );	// reset diff, rather than accumulating
					}
				}
			}
            if ( propagate_fidelity ) {
                gcm<DeviceMode>::mul( fidelity_conv_output_.count(), Omega, Omega_diff, Omega_diff );
                gcm<DeviceMode>::mul( fidelity_conv_output_.count(), Omega, Omega_diff, Omega_diff );
            }
		}
		// compute final diff and put them into normal ordered blob
		if ( this->param_propagate_down_[0] ) {	// filter

			Dtype* W_diff = weights_feature_->mutable_diff( DeviceMode() );
			const Dtype* W_tilde_diff = reordered_weights_.diff( DeviceMode() );
			int C = this->channels_, C_f = fidelity_channels_, R = fidelity_factor_,
					C_out = this->num_output_, K = patch_count_, N = this->num_;

            Dtype* dh_dw_bar_bar = buf_base;
            if (mean_removal_ && mean_adding_back_) {
			    const Dtype* X_bar = bottom_feature_mean_.data( DeviceMode() );
                gcm<DeviceMode>::gemm( CblasTrans, CblasNoTrans, C_out, C, N,
                        Dtype(1.), sample_bias_diff, X_bar, Dtype(0.), dh_dw_bar_bar );
            }

        	CommonFidaConvLayerAux<Dtype, DeviceMode>::filter_diff(
        			W_tilde_diff, dh_dw_bar_bar, W_diff, C, C_f, R, C_out, K, N,
        			this->reverse_dimensions(), mean_removal_ && mean_adding_back_ );

		}

		if ( propagate_fidelity ) {
            // accumulate diff from fidelity conv
			fidelity_conv_backward();
            //
			if (mean_removal_) {
				NOT_IMPLEMENTED;
			} else {
				if (!pre_scaled_) {	// accumulate diff from input scaling
					int N = this->num_, C_f = fidelity_channels_, G = bottom_geo_count_, R = fidelity_factor_;
					const Dtype* Z_diff = reordered_whittened_feature_.diff( DeviceMode() );
					const Dtype* X  = bottom_feature_->data( DeviceMode() );
					Dtype* Phi_diff = bottom_fidelity_->mutable_diff( DeviceMode() );
					CommonFidaConvLayerAux<Dtype, DeviceMode>::input_fidelity_diff_scaling(
							Z_diff, X, Phi_diff, N, C_f, G, R);
					// for mean_removal_ case, X should be mean removed.
				}
			}

		}

		if ( propagate_down[0] ) {	// input feature
			Dtype* X_diff       = bottom_feature_->mutable_diff( DeviceMode() );
			const Dtype* Z_diff = reordered_whittened_feature_.diff( DeviceMode() );
			int N = this->num_, C = this->channels_, C_f = fidelity_channels_,
					G = bottom_geo_count_, R = fidelity_factor_;

			{
				const Dtype* Phi = (pre_scaled_)?(NULL):(bottom_fidelity_->data( DeviceMode() ));
				CommonFidaConvLayerAux<Dtype, DeviceMode>::input_diff_scaling(
						Z_diff, Phi, X_diff, N, C, C_f, G, R );
			}

			if ( mean_removal_ ) {
				Dtype* dh_dX_bar = buf_base;	// N*C

				const Dtype* sum_mult_data = sum_mult_.data( DeviceMode() );

				gcm<DeviceMode>::gemv( CblasNoTrans, N*C, G, Dtype(-1.), X_diff, sum_mult_data,
						Dtype(0.), dh_dX_bar );
				if ( mean_adding_back_ ) {
					const Dtype* w_bar_bar = weights_sum_.data( DeviceMode() );
					int C_out = this->num_output_;
					gcm<DeviceMode>::gemm( CblasNoTrans, ((this->reverse_dimensions())?CblasTrans:CblasNoTrans),
							N, C, C_out,
							Dtype(1.), sample_bias_diff, w_bar_bar, Dtype(1.), dh_dX_bar  );
				}

				/*
				gcm<DeviceMode>::gemm( CblasNoTrans, CblasNoTrans, N*C, G, 1,
						Dtype(1.)/Dtype(G), dh_dX_bar, sum_mult_data,
						Dtype(1.), X_diff );
						*/
				const Dtype* normalized_bottom_fidelity_data = normalized_bottom_fidelity_.data( DeviceMode() );
				CommonFidaConvLayerAux<Dtype, DeviceMode>::input_diff_bar(
						dh_dX_bar, normalized_bottom_fidelity_data, X_diff,
						N, C, C_f, G );

			}
		}


	}

}


template <typename Dtype,int ConvOrDeconv_T>
void CommonFidaConvLayer<Dtype,ConvOrDeconv_T>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
	Forward_generic<mCPU>(bottom,top);
}


template <typename Dtype,int ConvOrDeconv_T>
void CommonFidaConvLayer<Dtype,ConvOrDeconv_T>::Backward_cpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	Backward_generic<mCPU>(top,propagate_down,bottom);
}


#ifndef CPU_ONLY

template <typename Dtype,int ConvOrDeconv_T>
void CommonFidaConvLayer<Dtype,ConvOrDeconv_T>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
	Forward_generic<mGPU>(bottom,top);
}


template <typename Dtype,int ConvOrDeconv_T>
void CommonFidaConvLayer<Dtype,ConvOrDeconv_T>::Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	Backward_generic<mGPU>(top,propagate_down,bottom);
}

#else

template <typename Dtype,int ConvOrDeconv_T>
void CommonFidaConvLayer<Dtype,ConvOrDeconv_T>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
	NO_GPU;
}


template <typename Dtype,int ConvOrDeconv_T>
void CommonFidaConvLayer<Dtype,ConvOrDeconv_T>::Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	NO_GPU;
}

#endif

INSTANTIATE_CLASS(FidaConvLayer);
INSTANTIATE_CLASS(FidaDeconvLayer);

REGISTER_LAYER_CLASS(FidaConv);
REGISTER_LAYER_CLASS(FidaDeconv);



}
