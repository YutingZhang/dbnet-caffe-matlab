/*
 * argmax_one_layer.cpp
 *
 *  Created on: Sep 20, 2015
 *      Author: zhangyuting
 */

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/argmax_one_layer.hpp"

using namespace std;

namespace caffe {

template <typename Dtype>
void ArgMaxOneLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void ArgMaxOneLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top) {
  axis_ = bottom[0]->CanonicalAxisIndex(this->layer_param_.argmax_one_param().axis());
  outer_num_ = bottom[0]->count(0, axis_);
  inner_num_ = bottom[0]->count(axis_ + 1);
  channel_num_ = bottom[0]->shape(axis_);
  auto top_shape = bottom[0]->shape();
  if (top_shape.size()>1) {
	  top_shape.erase( top_shape.begin()+axis_ );
  } else {
	  top_shape[0] = 1;
  }
  top[0]->Reshape( top_shape );
}

template <typename Dtype>
void ArgMaxOneLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	for ( int i=0; i<outer_num_ ; ++i) {
		const Dtype* datah = bottom_data + (i*channel_num_*inner_num_);
		Dtype*       idxh  = top_data + i*inner_num_;
		for ( int j=0; j<inner_num_; ++j ) {
			const Dtype* dataq = datah+j;
			int maxidx = -1; Dtype maxval = -FLT_MAX;
			for ( int c=0; c<channel_num_; ++c ) {
				if ( *dataq>maxval ) {
					maxval = *dataq;
					maxidx = c;
				}
				dataq += inner_num_;
			}
			idxh[j] = static_cast<Dtype>(maxidx);
		}
	}
}

template <typename Dtype>
void ArgMaxOneLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	// no backward
}

#ifdef CPU_ONLY
STUB_GPU(ArgMaxOneLayer);
#endif

INSTANTIATE_CLASS(ArgMaxOneLayer);
REGISTER_LAYER_CLASS(ArgMaxOne);

}  // namespace caffe
