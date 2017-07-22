#include <vector>

#include "caffe/layers/misc_seg_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void BasePrefetchingDataLayer_Fallback<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  JoinPrefetchThread();
  // Copy the data
  caffe_copy(prefetch_data_.count(), prefetch_data_.cpu_data(),
      top[0]->mutable_gpu_data());
  if (this->output_labels_) {
    caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
        top[1]->mutable_gpu_data());
  }
  // Start a new prefetch thread
  CreatePrefetchThread();
}

  /*
   * Jay add
   *
   */
/*
 notice:
 this code is based on the following implementation.
 https://bitbucket.org/deeplab/deeplab-public/
 */
template <typename Dtype>
void ImageDimPrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  BasePrefetchingDataLayer_Fallback<Dtype>::JoinPrefetchThread();
  // Copy the data
  caffe_copy(this->prefetch_data_.count(), this->prefetch_data_.cpu_data(),
             top[0]->mutable_gpu_data());
  if (this->output_labels_) {
    caffe_copy(this->prefetch_label_.count(), this->prefetch_label_.cpu_data(),
               top[1]->mutable_gpu_data());
  }
  if (output_data_dim_) {
    caffe_copy(prefetch_data_dim_.count(), prefetch_data_dim_.cpu_data(),
               top[2]->mutable_gpu_data());
  }

  // Start a new prefetch thread
  BasePrefetchingDataLayer_Fallback<Dtype>::CreatePrefetchThread();
}

INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer_Fallback);
INSTANTIATE_LAYER_GPU_FORWARD(ImageDimPrefetchingDataLayer);

}  // namespace caffe
