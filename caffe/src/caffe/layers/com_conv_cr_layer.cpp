#if 0

#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include <limits>

namespace caffe {

template <typename Dtype,int ConvOrDeconv_T>
void CommonConvolutionCRLayer<Dtype,ConvOrDeconv_T>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	CHECK_EQ( this->layer_param().convolution_param().group(), 1 ) << "It only supports group == 1";

	ch_propagate_down_vec_.clear(); ch_propagate_down_vec_.push_back(true);

	set_up_top_bottom(bottom,top);

	base_layer_t::LayerSetUp( my_bottom_, my_top_ );

    this->num_=0;  // guarantee Reshape can be triggered

	compute_rw_once_ = false; ever_computed_rw_ = false;

}

template <typename Dtype,int ConvOrDeconv_T>
void CommonConvolutionCRLayer<Dtype,ConvOrDeconv_T>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    
	if ( bottom.size()==1 ) {
		if (!default_rw_input_.get() ||
				bottom[0]->num()!=default_rw_input_->num() ||
				bottom[0]->height()!=default_rw_input_->height() ||
				bottom[0]->width()!=default_rw_input_->width() ) {
			default_rw_input_.reset( new Blob<Dtype>(bottom[0]->num(),1,bottom[0]->height(),bottom[0]->width()) );
			Dtype*  rw_data_in = default_rw_input_->mutable_cpu_data();
			caffe_set( default_rw_input_->count(), Dtype(1.), rw_data_in );
			compute_rw_once_ = true;
		}
	} else {
		CHECK( (bottom[0]->num()==bottom[1]->num() &&
			bottom[0]->height()==bottom[1]->height() &&
			bottom[0]->width()==bottom[1]->width()) ) <<
					"the two bottom blobs should have the same shape (expect for channels)";
		compute_rw_once_ = this->layer_param_.convolution_cr_param().compute_weights_once();
	}
	set_up_top_bottom(bottom,top);

    bool reshape_rw = false, reshape_ch = false;
    if (this->num_!=my_bottom_[0]->num() || this->height_!=my_bottom_[0]->height()
            || this->width_!=my_bottom_[0]->width())
        reshape_rw=reshape_ch=true;
    else {
        if (this->channels_!=my_bottom_[0]->channels()) reshape_ch = true;
        if (this->rw_channels0_!=my_weights_[0]->channels()) reshape_rw = true;
    }

	// set up the internal conv layer
	if (reshape_ch) {

	    base_layer_t::Reshape( my_bottom_, my_top_ );

		LayerParameter param;
		ConvolutionParameter* conv_param = param.mutable_convolution_param();
        *conv_param = this->layer_param().convolution_param();
		conv_param->set_bias_term( false );
		ch_conv_layer_.reset( new base_layer_t(param) );

		ch_input_.Reshape( my_bottom_[0]->num(), 1, my_bottom_[0]->height(), my_bottom_[0]->width() );
	    ch_bottom_.clear(); ch_top_.clear();
		ch_bottom_.push_back( &ch_input_ );
		ch_top_.push_back( &ch_output_ );
		ch_conv_layer_->SetUp(ch_bottom_, ch_top_);
		//ch_conv_layer_->Reshape(ch_bottom_, ch_top_);

		ch_weights_ = ch_conv_layer_->blobs()[0];

    	// init count
	    inHW_  = this->height_*this->width_;
    	outHW_ = this->height_out_*this->width_out_;
    	kernelCnt_   = this->kernel_h_*this->kernel_w_;
    	chWeightCnt_ = this->num_output_*kernelCnt_;
    	outCnt_ = ch_output_.count(); // same as my_top_[0]->count();

	}

	// set up re-weighting conv layer
	if (reshape_rw) {

		LayerParameter param;
		ConvolutionParameter* conv_param = param.mutable_convolution_param();
        *conv_param = this->layer_param().convolution_param();
		FillerParameter* weight_filler = conv_param->mutable_weight_filler();
		FillerParameter* bias_filler = conv_param->mutable_bias_filler();
		weight_filler->set_type( "constant" );
		weight_filler->set_value( Dtype(1.) );
		bias_filler->set_type( "constant" );
		bias_filler->set_value( Dtype(0.) );
		conv_param->set_num_output( 1 );
		rw_conv_layer_.reset( new base_layer_t(param) );

		rw_input_.Reshape( my_weights_[0]->num(), 1, my_weights_[0]->height(), my_weights_[0]->width() );
	    rw_bottom_.clear(); rw_top_.clear();
		rw_bottom_.push_back( &rw_input_ );
		rw_top_.push_back( &rw_output_ );
		rw_conv_layer_->SetUp(rw_bottom_, rw_top_);
		//rw_conv_layer_->Reshape(rw_bottom_, rw_top_);

		rw_final_.Reshape( rw_output_.num(), my_weights_[0]->channels(),
				rw_output_.height(), rw_output_.width() ); // permuted channel and num

        // init count
		ever_computed_rw_ = false;
		rw_channels_ = rw_channels0_ = my_weights_[0]->channels();

	}

}

template <typename Dtype,int ConvOrDeconv_T>
void CommonConvolutionCRLayer<Dtype,ConvOrDeconv_T>::set_up_top_bottom (
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	if (my_bottom_.size()) my_bottom_[0] = bottom[0];
	else my_bottom_.push_back(bottom[0]);

	if (my_top_.size()) my_top_[0] = top[0];
	else my_top_.push_back(top[0]);

	if (bottom.size()>1)
		if (my_weights_.size()) my_weights_[0] = bottom[1];
		else my_weights_.push_back(bottom[1]);
	else
		if (my_weights_.size()) my_weights_[0] = default_rw_input_.get();
		else my_weights_.push_back(default_rw_input_.get());

    //LOG(INFO) << "**** My Bottom: " << my_bottom_.size();

}


template <typename Dtype,int ConvOrDeconv_T>
void CommonConvolutionCRLayer<Dtype,ConvOrDeconv_T>::compute_weighting_map (
		const Dtype* bottom_weights, Dtype* ch_weights_in,
		Dtype* ch_weights_out, Dtype* final_weights ) {
	if (compute_rw_once_ && ever_computed_rw_) return;
    // LOG(INFO) << "*************** wm";
	bool channel_identical = compute_rw_once_;
	for ( int c = 0; c<rw_channels_; ++c ) {
		// copy data
		zcaffe_blockcopy( this->num_, inHW_, my_weights_[0]->offset(1,0,0,0),
				bottom_weights+my_weights_[0]->offset(0,c,0,0), inHW_, ch_weights_in );
    	// forward
		rw_conv_layer_->Forward(rw_bottom_, rw_top_);
		// copy to final
		zcaffe_blockcopy( this->num_, outHW_, outHW_, ch_weights_out,
				rw_final_.offset(1,0,0,0), final_weights + rw_final_.offset(0,c,0,0) );
		if (channel_identical) {
			Dtype* ch0_weights_ptr = final_weights + rw_final_.offset(0,c,0,0);
			Dtype* ch_weights_out_ptr = ch_weights_out;
			for ( int n = 0; n<this->num_ && channel_identical; ++n ) {
				Dtype* final_weights_ptr = final_weights + rw_final_.offset(n,c,0,0);
				Dtype l1dist;
				if (Caffe::mode() == Caffe::GPU ) {
#ifndef CPU_ONLY
					caffe_gpu_sub<Dtype>( outHW_ , final_weights_ptr, ch0_weights_ptr, ch_weights_out_ptr);
					caffe_gpu_asum<Dtype>( outHW_, ch_weights_out_ptr, &l1dist );
#endif
				} else {
					caffe_sub<Dtype>( outHW_ , final_weights_ptr, ch0_weights_ptr, ch_weights_out_ptr);
					l1dist = caffe_cpu_asum<Dtype>( outHW_, ch_weights_out_ptr );
				}
				ch_weights_out_ptr += outHW_;
				channel_identical = channel_identical && (l1dist<std::numeric_limits<Dtype>::epsilon());
			}
		}
		// consolidate the reweighting map if possible
	}
    if (Caffe::mode() == Caffe::GPU ) {
#ifndef CPU_ONLY
	    zcaffe_gpu_safeinv<Dtype>(rw_final_.count(),final_weights,final_weights);
#endif
    } else {
	    zcaffe_cpu_safeinv<Dtype>(rw_final_.count(),final_weights,final_weights);
    }
    if (channel_identical) {
    	LOG(INFO) << "Conv/DeconvCR: reweighting is found to be the same for all channels";
    	rw_channels_ = 1;
    }
	ever_computed_rw_ = true;
}


template <typename Dtype,int ConvOrDeconv_T>
void CommonConvolutionCRLayer<Dtype,ConvOrDeconv_T>::setup_channel_data_and_weights (
		int c, const Dtype* bottom_data, Dtype* ch_data_in,
		const Dtype* filter_data, Dtype* ch_filter_data ) {
    //LOG(INFO) << "cdw - c : " << c;
	// copy data
	zcaffe_blockcopy( this->num_, inHW_, my_bottom_[0]->offset(1,0,0,0),
			bottom_data + my_bottom_[0]->offset(0,c,0,0), inHW_, ch_data_in);
	// copy weights
	if (this->reverse_dimensions()) {
		const Dtype* filter_data_ptr = filter_data + this->blobs()[0]->offset(c,0,0,0);
		caffe_copy( chWeightCnt_, filter_data_ptr, ch_filter_data );
	} else {
		zcaffe_blockcopy( this->num_output_, kernelCnt_, this->blobs()[0]->offset(1,0,0,0),
				filter_data + this->blobs()[0]->offset(0,c,0,0), kernelCnt_, ch_filter_data);
	}
}

template <typename Dtype,int ConvOrDeconv_T>
void CommonConvolutionCRLayer<Dtype,ConvOrDeconv_T>::compute_channel_reweighted_conv (
		int c, const Dtype* final_weights, Dtype* ch_data_out, Dtype* top_data ) {

    //LOG(INFO) << "crc - c : " << c;
	// forward
	ch_conv_layer_->Forward(ch_bottom_, ch_top_);
    
	// reweighting
	int rw_c = c % rw_channels_;
	const Dtype* ch_data_out_ptr = ch_data_out;
	Dtype* top_data_ptr = top_data;
	const int el_stride = outHW_*this->num_output_;
	for ( int n = 0; n<this->num_; ++n ) {
		const Dtype* final_weights_ptr = final_weights + rw_final_.offset(n,rw_c,0,0);
		zcaffe_repmul( outHW_, final_weights_ptr, this->num_output_, ch_data_out_ptr, top_data_ptr, true );
		ch_data_out_ptr += el_stride; top_data_ptr += el_stride;
	}

}

template <typename Dtype,int ConvOrDeconv_T>
void CommonConvolutionCRLayer<Dtype,ConvOrDeconv_T>::setup_channel_reweighted_diff(
		int c, const Dtype* final_weights,
		const Dtype* top_diff, Dtype* ch_diff_out ) {
	//LOG(INFO) << "crd - c : " << c;
    int rw_c = c % rw_channels_;

    const Dtype* top_diff_ptr = top_diff;
    Dtype* ch_diff_out_ptr = ch_diff_out;
    const int el_stride = outHW_*this->num_output_;
	for ( int n = 0; n<this->num_; ++n ) {
		const Dtype* final_weights_ptr = final_weights + rw_final_.offset(n,rw_c,0,0);
		zcaffe_repmul( outHW_, final_weights_ptr, this->num_output_, top_diff_ptr, ch_diff_out_ptr );
		ch_diff_out_ptr += el_stride; top_diff_ptr += el_stride;
	}

}

template <typename Dtype,int ConvOrDeconv_T>
void CommonConvolutionCRLayer<Dtype,ConvOrDeconv_T>::compute_channel_conv_diff( int c,
		const Dtype* ch_diff_in, Dtype* ch_filter_diff, Dtype* bottom_diff, Dtype* filter_diff) {

	//LOG(INFO) << "ccd - c : " << c;
	if (Caffe::mode() == Caffe::GPU ) {
#ifndef CPU_ONLY
		caffe_gpu_set( chWeightCnt_, Dtype(0.), ch_filter_diff );
#endif
	} else {
		caffe_set( chWeightCnt_, Dtype(0.), ch_filter_diff );
	}
	ch_conv_layer_->Backward(ch_top_, ch_propagate_down_vec_, ch_bottom_);

	// copy data diff
	if (ch_propagate_down_vec_[0]) {
		zcaffe_blockcopy( this->num_, inHW_, inHW_, ch_diff_in,
				my_bottom_[0]->offset(1,0,0,0), bottom_diff + my_bottom_[0]->offset(0,c,0,0) );
	}
	// copy weights
	if (this->reverse_dimensions()) {
		Dtype* filter_diff_ptr = filter_diff + this->blobs()[0]->offset(c,0,0,0);
		zcaffe_axpy(chWeightCnt_, Dtype(1.), ch_filter_diff, filter_diff_ptr );	// accumulation instead of copy
	} else {
		zcaffe_blockaxpy( kernelCnt_, this->num_output_,
				kernelCnt_, Dtype(1.), ch_filter_diff,
				this->blobs()[0]->offset(1,0,0,0), filter_diff+this->blobs()[0]->offset(0,c,0,0) );
	}
}


template <typename Dtype,int ConvOrDeconv_T>
void CommonConvolutionCRLayer<Dtype,ConvOrDeconv_T>::Forward_generic(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	set_up_top_bottom(bottom,top);

	// compute reweighting mask
	{
        // input weighting:  all to single channel
		const Dtype* bottom_weights; Dtype* ch_weights_in;
        // output weighting: single channel to all
		Dtype* ch_weights_out, * final_weights;
		if (Caffe::mode() == Caffe::GPU ) {
#ifndef CPU_ONLY
			bottom_weights = my_weights_[0]->gpu_data();
			ch_weights_in = rw_input_.mutable_gpu_data();
			ch_weights_out = rw_output_.mutable_gpu_data();
			final_weights = rw_final_.mutable_gpu_data();
#endif
		} else {
			bottom_weights = my_weights_[0]->cpu_data();
			ch_weights_in = rw_input_.mutable_cpu_data();
			ch_weights_out = rw_output_.mutable_cpu_data();
			final_weights = rw_final_.mutable_cpu_data();
		}
        // computation
		compute_weighting_map( bottom_weights, ch_weights_in, ch_weights_out, final_weights );
	}

	// per-channel convolution and reweighting
    // accumulated reweighted conv output
	Dtype* top_data;
	{
        // input data:  all to single channel
		const Dtype* bottom_data;   Dtype* ch_data_in;
        // conv weights: all to single input channel
		const Dtype* filter_data;   Dtype* ch_filter_data;
        // reweighting single channel conv output
		const Dtype* final_weights; Dtype* ch_data_out;

		if (Caffe::mode() == Caffe::GPU ) {
#ifndef CPU_ONLY
			bottom_data = my_bottom_[0]->gpu_data();
			ch_data_in = ch_input_.mutable_gpu_data();
			filter_data = this->blobs()[0]->gpu_data();
			ch_filter_data = ch_weights_->mutable_gpu_data();
			final_weights = rw_final_.gpu_data();
			ch_data_out = ch_output_.mutable_gpu_data();
			top_data = my_top_[0]->mutable_gpu_data();
#endif
		} else {
			bottom_data = my_bottom_[0]->cpu_data();
			ch_data_in = ch_input_.mutable_cpu_data();
			filter_data = this->blobs()[0]->cpu_data();
			ch_filter_data = ch_weights_->mutable_cpu_data();
			final_weights = rw_final_.cpu_data();
			ch_data_out = ch_output_.mutable_cpu_data();
			top_data = my_top_[0]->mutable_cpu_data();
		}
		if (rw_channels_==1) {
			// compute the weights in one shot, as the weights are shared for all channels
			bool orig_bias_term = this->bias_term_;
			this->bias_term_ = false;
		    if (Caffe::mode() == Caffe::GPU ) {
#ifndef CPU_ONLY
                base_layer_t::Forward_gpu( my_bottom_, my_top_ );
#endif
            } else {
                base_layer_t::Forward_cpu( my_bottom_, my_top_ );
            }
			this->bias_term_ = orig_bias_term;

			// reweighting
			Dtype* top_data_ptr = top_data;
			const int el_stride = outHW_*this->num_output_;
			for ( int n = 0; n<this->num_; ++n ) {
				const Dtype* final_weights_ptr = final_weights + rw_final_.offset(n,0,0,0);
				zcaffe_repmul( outHW_, final_weights_ptr, this->num_output_, top_data_ptr, top_data_ptr );
				top_data_ptr += el_stride;
			}
		} else {
			// channel-wise computation
			zcaffe_set( outCnt_, Dtype(0), top_data );
			for ( int c = 0; c<this->channels_; ++c ) {
				setup_channel_data_and_weights( c, bottom_data, ch_data_in,
						filter_data, ch_filter_data );
				compute_channel_reweighted_conv( c, final_weights, ch_data_out, top_data );
			}
		}
	}
	// add bias
	if (Caffe::mode() == Caffe::GPU ) {
#ifndef CPU_ONLY
        if (this->bias_term_) {
          const Dtype* bias = this->blobs_[1]->gpu_data();
          for (int n = 0; n < this->num_; ++n) {
            this->forward_gpu_bias(top_data + my_top_[0]->offset(n), bias);
          }
        }
#endif
    } else {
        if (this->bias_term_) {
          const Dtype* bias = this->blobs_[1]->cpu_data();
          for (int n = 0; n < this->num_; ++n) {
            this->forward_cpu_bias(top_data + my_top_[0]->offset(n), bias);
          }
        }
    }
}

template <typename Dtype,int ConvOrDeconv_T>
void CommonConvolutionCRLayer<Dtype,ConvOrDeconv_T>::Backward_generic (
		const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {

	ch_propagate_down_vec_[0] = propagate_down[0];

	set_up_top_bottom(bottom,top);

    // input data:  all to single channel
	const Dtype* bottom_data; Dtype* ch_data_in;
    // conv weights: all to single input channel
	const Dtype* filter_data; Dtype* ch_filter_data;
	// diff reweighting and to channel-wise
	const Dtype* final_weights; Dtype * top_diff, * ch_diff_out;
	// diff: channel -> entire
	const Dtype* ch_diff_in; Dtype* ch_filter_diff, * bottom_diff, * filter_diff;

	if (Caffe::mode() == Caffe::GPU ) {
#ifndef CPU_ONLY
		bottom_data = my_bottom_[0]->gpu_data();
		ch_data_in  = ch_input_.mutable_gpu_data();
		filter_data = this->blobs()[0]->gpu_data();
		ch_filter_data = ch_weights_->mutable_gpu_data();
		final_weights  = rw_final_.gpu_data();
		top_diff    = my_top_[0]->mutable_gpu_diff();
		ch_diff_out = ch_output_.mutable_gpu_diff();
		ch_diff_in  = ch_input_.gpu_diff();
		ch_filter_diff = ch_weights_->mutable_gpu_diff();
		bottom_diff = my_bottom_[0]->mutable_gpu_diff();
		filter_diff = this->blobs()[0]->mutable_gpu_diff();
        // add bias
        if (this->bias_term_ && this->param_propagate_down_[1]) {
          Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff(); // Do I need to set it to zero first??
          for (int n = 0; n < this->num_; ++n) {
            this->backward_gpu_bias(bias_diff, top_diff + my_top_[0]->offset(n));
          }
        }
#endif
	} else {
		bottom_data = my_bottom_[0]->cpu_data();
		ch_data_in  = ch_input_.mutable_cpu_data();
		filter_data = this->blobs()[0]->cpu_data();
		ch_filter_data = ch_weights_->mutable_cpu_data();
		final_weights  = rw_final_.cpu_data();
		top_diff    = my_top_[0]->mutable_cpu_diff();
		ch_diff_out = ch_output_.mutable_cpu_diff();
		ch_diff_in  = ch_input_.cpu_diff();
		ch_filter_diff = ch_weights_->mutable_cpu_diff();
		bottom_diff = my_bottom_[0]->mutable_cpu_diff();
		filter_diff = this->blobs()[0]->mutable_cpu_diff();
        // add bias
        if (this->bias_term_ && this->param_propagate_down_[1]) {
          Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff(); // Do I need to set it to zero first??
          for (int n = 0; n < this->num_; ++n) {
            this->backward_cpu_bias(bias_diff, top_diff + my_top_[0]->offset(n));
          }
        }
	}

	if ( rw_channels_==1 ) {
	    Dtype* top_diff_ptr = top_diff;
	    const int el_stride = outHW_*this->num_output_;
		for ( int n = 0; n<this->num_; ++n ) {
			const Dtype* final_weights_ptr = final_weights + rw_final_.offset(n,0,0,0);
			zcaffe_repmul( outHW_, final_weights_ptr, this->num_output_, top_diff_ptr, top_diff_ptr );
			top_diff_ptr += el_stride;
		}

		bool orig_bias_term = this->bias_term_;
		this->bias_term_ = false;
        if (Caffe::mode() == Caffe::GPU ) {
#ifndef CPU_ONLY
            base_layer_t::Backward_gpu( my_top_, ch_propagate_down_vec_, my_bottom_ );
#endif
        } else {
            base_layer_t::Backward_cpu( my_top_, ch_propagate_down_vec_, my_bottom_ );
        }
		this->bias_term_ = orig_bias_term;
	} else {
		// channel-wise computation
		for ( int c = 0; c<this->channels_; ++c ) {
			setup_channel_data_and_weights( c, bottom_data, ch_data_in,
					filter_data, ch_filter_data );
			// ch_conv_layer_->Forward(ch_bottom_, ch_top_);
			setup_channel_reweighted_diff( c, final_weights, top_diff, ch_diff_out );
			compute_channel_conv_diff( c, ch_diff_in, ch_filter_diff,
					bottom_diff, filter_diff );
		}
	}

}

template <typename Dtype,int ConvOrDeconv_T>
void CommonConvolutionCRLayer<Dtype,ConvOrDeconv_T>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	Forward_generic( bottom, top );
}

template <typename Dtype,int ConvOrDeconv_T>
void CommonConvolutionCRLayer<Dtype,ConvOrDeconv_T>::Backward_cpu(
		const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
	Backward_generic( top, propagate_down, bottom );
}


#ifndef CPU_ONLY

template <typename Dtype,int ConvOrDeconv_T>
void CommonConvolutionCRLayer<Dtype,ConvOrDeconv_T>::Forward_gpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	Forward_generic( bottom, top );
}

template <typename Dtype,int ConvOrDeconv_T>
void CommonConvolutionCRLayer<Dtype,ConvOrDeconv_T>::Backward_gpu(
		const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
	Backward_generic( top, propagate_down, bottom );
}

#endif

/*
#ifdef CPU_ONLY
STUB_GPU(ConvolutionCRLayer);
STUB_GPU(DeconvolutionCRLayer);
#endif
*/

INSTANTIATE_CLASS(ConvolutionCRLayer);
INSTANTIATE_CLASS(DeconvolutionCRLayer);

REGISTER_LAYER_CLASS(ConvolutionCR);
REGISTER_LAYER_CLASS(DeconvolutionCR);

}  // namespace caffe

#endif

