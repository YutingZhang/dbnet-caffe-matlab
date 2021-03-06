#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/upsampling_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void UpsampleForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, const int upsampled_height, const int upsampled_width,
    const int kernel_h, const int kernel_w, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int uw = index % upsampled_width;
    int uh = (index / upsampled_width) % upsampled_height;
    int c = (index / upsampled_width / upsampled_height) % channels;
    int n = index / upsampled_width / upsampled_height / channels;
    bottom_data += (n * channels + c) * height * width;
    if (uw % kernel_w == 0 && uh % kernel_h == 0) {
	int sw = static_cast<int>(static_cast<float>(uw) / kernel_w); 
	int sh = static_cast<int>(static_cast<float>(uh) / kernel_h);
    	top_data[index] = bottom_data[sh * width + sw];
    }
    else {
	top_data[index] = 0;
    }
  }
}


template <typename Dtype>
void UpsamplingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
  UpsampleForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, upsampled_height_, upsampled_width_, kernel_h_, kernel_w_, top_data);
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void UpsampleBackward(const int nthreads, const Dtype* top_diff,
    const int num, const int channels, const int height,
    const int width, const int upsampled_height, const int upsampled_width,
    const int kernel_h, const int kernel_w, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    top_diff += (n * channels + c) * upsampled_height * upsampled_width;
    int uw = static_cast<int>(static_cast<float>(w) * kernel_w);
    int uh = static_cast<int>(static_cast<float>(h) * kernel_h);
    bottom_diff[index] = top_diff[uh * upsampled_width + uw];
  }
}


template <typename Dtype>
void UpsamplingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  // NOLINT_NEXT_LINE(whitespace/operators)
  UpsampleBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, top[0]->num(), channels_,
      height_, width_, upsampled_height_, upsampled_width_, kernel_h_, kernel_w_, bottom_diff);
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(UpsamplingLayer);


}  // namespace caffe
