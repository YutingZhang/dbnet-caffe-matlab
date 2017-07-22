/*
 * eltwise_with_filler_layer.cpp
 *
 *  Created on: Sep 20, 2015
 *      Author: zhangyuting
 */

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/eltwise_with_filler_layer.hpp"
#include "caffe/layers/dummy_data_layer.hpp"
#include "caffe/layers/eltwise_layer.hpp"

using namespace std;

namespace caffe {

template <typename Dtype>
void EltwiseWithFillerLayer<Dtype>::setup_internal_bottom_top(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top) {
	dummy_bottom_.clear();
	if ( dummy_top_.empty() )
		dummy_top_.push_back( filler_cache_.get() );
	else
		dummy_top_[0] = filler_cache_.get();
	if ( eltwise_bottom_.empty() ) {
		eltwise_bottom_.push_back( bottom[0] );
		eltwise_bottom_.push_back( filler_cache_.get() );
	} else {
		eltwise_bottom_[0] = bottom[0];
		eltwise_bottom_[1] = filler_cache_.get();
	}
	eltwise_top_ = top;
}

template <typename Dtype>
void EltwiseWithFillerLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top) {
	const EltwiseWithFillerParameter& ewf_param = this->layer_param().eltwise_with_filler_param();

	auto& bottom_shape = bottom[0]->shape();
	filler_cache_.reset( new Blob<Dtype>(bottom_shape) );
	{
		LayerParameter dummy_layer_param;
		dummy_layer_param.set_name( string( this->layer_param().name() ) + " [dummy_data]" );
		dummy_layer_param.set_type( "DummyData" );
		auto dummy_data_param = dummy_layer_param.mutable_dummy_data_param();
		*( dummy_data_param->add_data_filler() ) = ewf_param.filler();
		auto dummy_data_shape = dummy_data_param->add_shape();
		dummy_data_shape->clear_dim();
		for ( auto s :  bottom_shape )
			dummy_data_shape->add_dim(s);
		// dummy_data_layer_.reset( new DummyDataLayer<Dtype>(dummy_layer_param) );
		dummy_data_layer_ = dynamic_pointer_cast< DummyDataLayer<Dtype> > (
				LayerRegistry<Dtype>::CreateLayer(dummy_layer_param) );
	}
	{
		LayerParameter eltwise_layer_param;
		eltwise_layer_param.set_name( string( this->layer_param().name() ) + " [eltwise]" );
		eltwise_layer_param.set_type( "Eltwise" );
		*( eltwise_layer_param.mutable_eltwise_param() ) = ewf_param.eltwise();
		// eltwise_layer_.reset( new EltwiseLayer<Dtype>(eltwise_layer_param) );
		eltwise_layer_ = dynamic_pointer_cast< EltwiseLayer<Dtype> >(
				LayerRegistry<Dtype>::CreateLayer(eltwise_layer_param) );
	}
	eltwise_propagate_down_.clear();
	eltwise_propagate_down_.push_back( true );
	eltwise_propagate_down_.push_back( false );
	// Setup sublayers
	setup_internal_bottom_top( bottom, top );
	dummy_data_layer_->SetUp( dummy_bottom_, dummy_top_ );
	eltwise_layer_->SetUp( eltwise_bottom_, eltwise_top_ );

	channels_ = 0; inner_count_ = 0; outer_count_ = 0;
}

template <typename Dtype>
void EltwiseWithFillerLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top) {

	auto& bottom_shape = bottom[0]->shape();
	filler_cache_.reset( new Blob<Dtype>(bottom_shape) );
	setup_internal_bottom_top( bottom, top );
	dummy_data_layer_->Reshape( dummy_bottom_, dummy_top_ );
	eltwise_layer_->Reshape( eltwise_bottom_, eltwise_top_ );

	if ( bottom.size()>=2 ) {
		CHECK_LE( bottom[1]->num_axes(), bottom[0]->num_axes() )
				<< "scaling input should not have more axes than the data input";
		inner_count_ = 1;
		outer_count_ = 1;
		channels_ = 1;
		for ( int i = 0; i<bottom[0]->num_axes(); ++i ) {
			if ( i >= bottom[1]->num_axes() )
				inner_count_ *= bottom[0]->shape(i);
			else if ( bottom[1]->shape(i) == 1 ) {
				if (channels_ == 1)
					outer_count_ *= bottom[0]->shape(i);
				else
					inner_count_ *= bottom[0]->shape(i);
			} else {
				CHECK_GE( bottom[1]->shape(i), 1 ) << "Do not support empty channel";
				CHECK_EQ( bottom[1]->shape(i), bottom[0]->shape(i) ) << "channel dim must be matched";
				CHECK( channels_ == 1 ) << "Do not support multiple channels";
				channels_ = bottom[1]->shape(i);
			}
		}
		if (channels_==1) {
			inner_count_ *= outer_count_;
			outer_count_  = 1;
		}
	}

}

template <typename Dtype>
template <class DeviceMode>
void EltwiseWithFillerLayer<Dtype>::Forward_generic(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top) {
	setup_internal_bottom_top( bottom, top );
	dummy_data_layer_->Forward( dummy_bottom_, dummy_top_ );
	if ( bottom.size() >= 2 ) {
		Dtype* filler_data = dummy_top_[0]->mutable_data( DeviceMode() );
		const Dtype* scal_data = bottom[1]->data( mCPU() );
		for ( int i=0; i<outer_count_; ++i ) {
            for ( int j=0; j<channels_; ++j ) {
                Dtype a = std::sqrt( scal_data[j] );
                gcm<DeviceMode>::scal( inner_count_, a, filler_data );
                filler_data += inner_count_;
            }
		}
	}
	eltwise_layer_->Forward( eltwise_bottom_, eltwise_top_ );
}

template <typename Dtype>
void EltwiseWithFillerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top) {
	Forward_generic<mCPU>( bottom, top );
}

template <typename Dtype>
void EltwiseWithFillerLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if ( propagate_down[0] ) {
		setup_internal_bottom_top( bottom, top );
		eltwise_layer_->Backward( eltwise_top_, eltwise_propagate_down_, eltwise_bottom_ );
	}
}


#ifdef CPU_ONLY

STUB_GPU(EltwiseWithFillerLayer);

#else

template <typename Dtype>
void EltwiseWithFillerLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top) {
	Forward_generic<mGPU>( bottom, top );
}

template <typename Dtype>
void EltwiseWithFillerLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    Backward_cpu( top, propagate_down, bottom );
}

#endif

INSTANTIATE_CLASS(EltwiseWithFillerLayer);
REGISTER_LAYER_CLASS(EltwiseWithFiller);

}  // namespace caffe
