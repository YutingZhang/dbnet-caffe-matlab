#ifndef CAFFE_BN_LAYER_HPP_
#define CAFFE_BN_LAYER_HPP_

#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class BNLayer : public Layer<Dtype> {
 /*
 notice:
 this code is based on the implementation of by following authors.

 ducha-aiki: https://github.com/ducha-aiki
 ChenglongChen: https://github.com/ChenglongChen
 Russell91: https://github.com/Russell91
 jjkjkj: https://github.com/jjkjkj

 detailed discussion of this implementation can be found at:
 https://github.com/BVLC/caffe/pull/1965
 */
 public:
  explicit BNLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "BN"; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual int MaxBottomBlobs() const;
  virtual int MaxTopBlobs() const;

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  template<class DeviceMode>
  void Forward_generic(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);
  template<class DeviceMode>
  void Backward_generic(const vector<Blob<Dtype>*>& top,
	  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


  int bn_mode_;

  // spatial mean & variance
  Blob<Dtype> spatial_mean_, spatial_variance_;
  // batch mean & variance
  Blob<Dtype> batch_mean_, batch_variance_;
  // buffer blob
  DynamicBlobAt<Dtype,SyncedMemoryAllocator::LARGE> buffer_blob_;

  // merged scale and shift
  Blob<Dtype> mscale_, mshift_;
  Blob<Dtype> std_buffer_;

  Blob<Dtype> x_norm_;
  // x_sum_multiplier is used to carry out sum using BLAS
  Blob<Dtype> spatial_sum_multiplier_, batch_sum_multiplier_;

  // dimension
  int axis_;
  int N_;
  int C_;
  int G_;	// geometric number
  int cnt_;
  // eps
  Dtype var_eps_;
};

}

#endif

