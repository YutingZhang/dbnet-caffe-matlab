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
#include "caffe/util/math_functions.cuh"

using namespace std;

namespace caffe {

template <typename Dtype>
__global__
void ArgMaxOneLayerKernel(const int count,
		const Dtype* val_data, const int inner_num, const int channel_num, Dtype* midx_data ) {
  CUDA_KERNEL_LOOP(index, count) {
	  int i = index/inner_num;	// outer index
	  int j = index%inner_num;	// inner index
	  const Dtype* dataq = val_data  + i*channel_num*inner_num + j;
	  int maxidx = -1; Dtype maxval = -FLT_MAX;
	  for ( int c=0; c<channel_num; ++c ) {
		  if ( *dataq>maxval ) {
			maxval = *dataq;
			maxidx = c;
		  }
		  dataq += inner_num;
	  }
	  midx_data[index] = static_cast<Dtype>(maxidx);
  }
}

template <typename Dtype>
void ArgMaxOneLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top) {
	const int n = outer_num_*inner_num_;
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	ArgMaxOneLayerKernel<Dtype><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>( n,
					bottom_data, inner_num_, channel_num_, top_data );
}

template <typename Dtype>
void ArgMaxOneLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	// No backward
}

INSTANTIATE_LAYER_GPU_FUNCS(ArgMaxOneLayer);

}  // namespace caffe
