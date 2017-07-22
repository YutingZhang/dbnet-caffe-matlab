#ifndef CAFFE_ARGMAX_ONE_LAYER_HPP_
#define CAFFE_ARGMAX_ONE_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "boost/shared_ptr.hpp"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

template <typename Dtype>
class ArgMaxOneLayer : public Layer<Dtype> {
 public:
  explicit ArgMaxOneLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top);
  virtual inline const char*  type() const { return "ArgMaxOne"; }
  virtual inline int ExactTopBlobs() const { return 1; }
  virtual inline int ExactBottomBlobs() const { return 1; }
  //virtual inline bool EqualNumBottomTopBlobs() const { return ture; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 protected:
  int axis_;
  int outer_num_, inner_num_, channel_num_;
};

}

#endif

