#ifndef CAFFE_RUNNING_AVERAGE_LAYER_HPP_
#define CAFFE_RUNNING_AVERAGE_LAYER_HPP_

#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class RunningAverageLayer : public Layer<Dtype> {
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
  explicit RunningAverageLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "RunningAverage"; }

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


  Dtype decay_;
  bool is_init_iter_;
};

}

#endif

