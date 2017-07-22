#include <algorithm>
#include <vector>

#include "caffe/layers/running_average_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <map>
#include <boost/thread/mutex.hpp>

namespace caffe {

using namespace std;


template<typename Dtype>
void RunningAverageLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  for (int i = 0; i < bottom.size(); ++i) {
    top[i]->ReshapeLike(*bottom[i]);
    top[i]->ShareData(*this->blobs_[i]);	// do not share diff
  }

}

template<typename Dtype>
void RunningAverageLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  Dtype mass = this->layer_param_.running_average_param().mass();
  decay_ = mass / (Dtype(1.)+mass);


  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(bottom.size()+1);
    for (int i = 0; i < bottom.size(); ++i) {
      this->blobs_[i].reset(new Blob<Dtype>(bottom[i]->shape()));
    }
    this->blobs_[bottom.size()].reset(new Blob<Dtype>(1, 1, 1, 1)); // weights
    caffe_set(this->blobs_[bottom.size()]->count(), Dtype(0),
                this->blobs_[bottom.size()]->mutable_cpu_data());
  }

  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), false);

  //
  is_init_iter_ = true;
}

template <typename Dtype>
template <class DeviceMode>
void RunningAverageLayer<Dtype>::Forward_generic(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

  if (this->phase_ == TRAIN || this->layer_param_.running_average_param().update_test() ) {

	bool reset_first = this->layer_param_.running_average_param().reset_history() && is_init_iter_;
	is_init_iter_ = false;

	Dtype& weight = *(this->blobs_[bottom.size()]->mutable_cpu_data());
	if (reset_first) weight = Dtype(0);

    for (int i = 0; i < bottom.size(); ++i) {
      const Dtype* bottom_data = bottom[i]->data( DeviceMode() );
      Dtype* mean_data = this->blobs_[i]->mutable_data( DeviceMode() );
      const int N = this->blobs_[i]->count();

      //if (reset_first) caffe_set(N, Dtype(0), mean_data);

      // m(t+1)=(m(t)*w(t)*a+x(t+1))/w(t+1)
      gcm<DeviceMode>::scal(N, weight*decay_, mean_data);
      gcm<DeviceMode>::add(N, bottom_data, mean_data, mean_data);
      gcm<DeviceMode>::scal(N, 1/(weight*decay_+1), mean_data);

      // blobs_ and top_ are shared
      //Dtype* top_data = top[i]->mutable_data( DeviceMode() );
      // top <- mean
      // caffe_copy(N, mean_data, top_data);
    }

    weight = weight*decay_ + 1;
  } else {
	  // Do nothing
	  // blobs_ and top_ are shared
	  /*
    for (int i = 0; i < bottom.size(); ++i) {
      Dtype* top_data = top[i]->mutable_cpu_data();
      Dtype* mean_data = this->blobs_[i]->mutable_cpu_data();
      const int N = this->blobs_[i]->count();

      // top <- mean
      caffe_copy(N, mean_data, top_data);
    }
    */
  }

}

template <typename Dtype>
template <class DeviceMode>
void RunningAverageLayer<Dtype>::Backward_generic(const vector<Blob<Dtype>*>& top,
	  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

template<typename Dtype>
void RunningAverageLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	Forward_generic<mCPU>(bottom,top);
}

template<typename Dtype>
void RunningAverageLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
	Backward_generic<mCPU>(top,propagate_down,bottom);
}

#ifdef CPU_ONLY
template<typename Dtype>
void RunningAverageLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	NO_GPU;
}

template<typename Dtype>
void RunningAverageLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
	NO_GPU;
}
#else

template<typename Dtype>
void RunningAverageLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	Forward_generic<mGPU>(bottom,top);
}

template<typename Dtype>
void RunningAverageLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
	Backward_generic<mGPU>(top,propagate_down,bottom);
}

#endif

INSTANTIATE_CLASS(RunningAverageLayer);
REGISTER_LAYER_CLASS(RunningAverage);

}  // namespace caffe
