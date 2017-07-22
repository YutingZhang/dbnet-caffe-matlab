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
void NaN2ZeroLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = (bottom_data[i] == bottom_data[i])? bottom_data[i] : (Dtype(0));
  }
}

template <typename Dtype>
void NaN2ZeroLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = ( top_diff[i] == top_diff[i] &&
    		  bottom_data[i] == bottom_data[i]) ? top_diff[i] : (Dtype(0));
      // the second condition, bottom_data[index] == bottom_data[index], should not be useful,
      // as bottom_data must be non-nan after feedforward
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(NaN2ZeroLayer);
#endif

INSTANTIATE_CLASS(NaN2ZeroLayer);
REGISTER_LAYER_CLASS(NaN2Zero);


}  // namespace caffe
