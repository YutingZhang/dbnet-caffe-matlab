/*
 * nan2zero_layer.cpp
 *
 *  Created on: May 17, 2014
 *      Author: zhangyuting
 */


#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/nan2zero_layer.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
__global__ void NaN2ZeroForward(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = (in[index] == in[index]) ? in[index] : 0;
  }
}

template <typename Dtype>
void NaN2ZeroLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  NaN2ZeroForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void NaN2ZeroBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = (in_diff[index] == in_diff[index] &&
    		in_data[index] == in_data[index])? in_diff[index] : 0;
    // the second condition, in_data[index] == in_data[index], should not be useful,
    // as in_data must be non-nan after feedforward
  }
}

template <typename Dtype>
void NaN2ZeroLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    NaN2ZeroBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(NaN2ZeroLayer);

}  // namespace caffe

