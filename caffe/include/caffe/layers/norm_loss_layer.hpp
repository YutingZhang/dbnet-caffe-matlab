#ifndef CAFFE_NORM_LOSS_LAYER_HPP_
#define CAFFE_NORM_LOSS_LAYER_HPP_

#include "boost/scoped_ptr.hpp"
#include "boost/shared_ptr.hpp"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class NormLossLayer : public LossLayer<Dtype> {
 public:
  explicit NormLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), buffer1_(), buffer2_() {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                     const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "NormLoss"; }
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }
  virtual inline int ExactNumBottomBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual Dtype normalize_denominator( int count, int batch_size );

  int norm_type_;
  Dtype p_;

  Blob<Dtype> buffer1_;
  Blob<Dtype> buffer2_;
};

}

#endif

