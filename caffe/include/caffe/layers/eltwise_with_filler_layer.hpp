#ifndef CAFFE_ELTWISE_WITH_FILLER_HPP_
#define CAFFE_ELTWISE_WITH_FILLER_HPP_

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

template <typename Dtype> class DummyDataLayer;
template <typename Dtype> class EltwiseLayer;

template <typename Dtype>
class EltwiseWithFillerLayer : public Layer<Dtype> {
public:
	  explicit EltwiseWithFillerLayer(const LayerParameter& param)
		 : Layer<Dtype>(param) {}
	  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			  const vector<Blob<Dtype>*>& top);
	  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			  const vector<Blob<Dtype>*>& top);
	  virtual inline const char*  type() const { return "EltwiseWithFiller"; }
	  virtual inline int ExactTopBlobs() const { return 1; }
	  virtual inline int MinBottomBlobs() const { return 1; }
	  virtual inline int MaxBottomBlobs() const { return 2; }	// the second one is for channel-wise re-scaling
	  //virtual inline bool EqualNumBottomTopBlobs() const { return true; }

	 protected:
	  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			  const vector<Blob<Dtype>*>& top);
	  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			  const vector<Blob<Dtype>*>& top);
	  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
	      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
	      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	  template <class DeviceMode>
	  void Forward_generic(const vector<Blob<Dtype>*>& bottom,
			  const vector<Blob<Dtype>*>& top);

	  void setup_internal_bottom_top(const vector<Blob<Dtype>*>& bottom,
			  const vector<Blob<Dtype>*>& top);

	 protected:
	  shared_ptr< Blob<Dtype> > filler_cache_;
	  shared_ptr< DummyDataLayer<Dtype> > dummy_data_layer_;
	  shared_ptr< EltwiseLayer<Dtype> >   eltwise_layer_;
	  vector< Blob<Dtype>* > dummy_bottom_;
	  vector< Blob<Dtype>* > dummy_top_;
	  vector< Blob<Dtype>* > eltwise_bottom_;
	  vector< Blob<Dtype>* > eltwise_top_;
	  vector<bool> eltwise_propagate_down_;

	  int inner_count_, outer_count_, channels_;
};

}

#endif

