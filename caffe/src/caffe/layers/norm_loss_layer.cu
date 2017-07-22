#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/norm_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void NormLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();

  Dtype loss = Dtype(0.);
  switch (norm_type_) {
  case NormLossParameter_NormType_L1:
  {
	  caffe_gpu_asum(count, bottom[0]->gpu_data(), &loss);
	  break;
  }
  case NormLossParameter_NormType_L2:
  {
      const Dtype* bottom_data = bottom[0]->gpu_data();
	  caffe_gpu_dot(count, bottom_data, bottom_data, &loss);
	  loss = Dtype(.5) * loss;
	  break;
  }
  case NormLossParameter_NormType_Lp:
  {
	  Dtype* buf1 = buffer1_.mutable_gpu_data();
	  Dtype* buf2 = buffer2_.mutable_gpu_data();
	  caffe_gpu_abs(count, bottom[0]->gpu_data(), buf2 );
	  caffe_gpu_powx(count, buf2, p_, buf1);
	  caffe_gpu_asum(count, buf1, &loss);
	  loss = loss / p_;
	  break;
  }
  default:
	  LOG(ERROR) << "NormLossLayer: Forward: Unknown norm_type_ : " << norm_type_;
  }
  loss /= this->normalize_denominator( count, bottom[0]->num() );
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void NormLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	int count = bottom[0]->count();
	const Dtype alpha = top[0]->cpu_diff()[0] /
			this->normalize_denominator( count, bottom[0]->num() );
    if (propagate_down[0]) {
	  switch (norm_type_) {
	  case NormLossParameter_NormType_L1:
	  {
          //LOG(INFO) << "S1";
		  Dtype* buf1 = buffer1_.mutable_gpu_data();
          //LOG(INFO) << "S2";
		  caffe_gpu_sign(count, bottom[0]->gpu_data(), buf1 );
          //LOG(INFO) << "S3";
	      caffe_gpu_axpby(
	          count,              // count
	          alpha,                           // alpha
			  buf1,                       // a
	          Dtype(0),                        // beta
	          bottom[0]->mutable_gpu_diff());  // b
          //LOG(INFO) << "S4";
		  break;
	  }
	  case NormLossParameter_NormType_L2:
	  {
	      caffe_gpu_axpby(
	          count,                           // count
	          alpha,                           // alpha
			  bottom[0]->gpu_data(),           // a
	          Dtype(0),                        // beta
	          bottom[0]->mutable_gpu_diff());  // b
		  break;
	  }
	  case NormLossParameter_NormType_Lp:
	  {
		  Dtype* buf1 = buffer1_.mutable_gpu_data();
		  Dtype* buf2 = buffer1_.mutable_gpu_data();
		  caffe_gpu_sign(count, buf2, buf2 );	// avoid div by 0
		  caffe_gpu_add(count, bottom[0]->gpu_data(), buf2, buf2);
		  caffe_gpu_div(count, buf1, buf2, buf1);
	      caffe_gpu_axpby(
	          count,                           // count
	          alpha,                           // alpha
			  buf1,           			       // a
	          Dtype(0),                        // beta
	          bottom[0]->mutable_gpu_diff());  // b
		  break;
	  }
	  default:
		  LOG(ERROR) << "NormLossLayer: Backward: Unknown norm_type_ : " << norm_type_;
	  }
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(NormLossLayer);

}  // namespace caffe
