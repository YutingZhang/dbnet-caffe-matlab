#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/norm_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void NormLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //LossLayer<Dtype>::Reshape(bottom, top);

  vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);

  buffer1_.ReshapeLike(*bottom[0]);
  buffer2_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void NormLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                     const vector<Blob<Dtype>*>& top) {

    LossLayer<Dtype>::LayerSetUp( bottom, top );

	norm_type_ = this->layer_param_.norm_loss_param().type();
	if ( norm_type_ == NormLossParameter_NormType_Lp ) {
		norm_type_ = NormLossParameter_NormType_Lp;
		p_ = this->layer_param_.norm_loss_param().p();
		CHECK_GE(p_,1) << "p must be >= 1";
		if (p_ == Dtype(1)) norm_type_ = NormLossParameter_NormType_L1;
        else if (p_ == Dtype(2)) norm_type_ = NormLossParameter_NormType_L2;
    }

}

template <typename Dtype>
void NormLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();

  Dtype loss = Dtype(0.);
  switch (norm_type_) {
  case NormLossParameter_NormType_L1:
  {
	  loss = caffe_cpu_asum(count, bottom[0]->cpu_data());
	  break;
  }
  case NormLossParameter_NormType_L2:
  {
      const Dtype* bottom_data = bottom[0]->cpu_data();
	  loss = caffe_cpu_dot(count, bottom_data, bottom_data);
	  loss = Dtype(.5) * loss;
	  break;
  }
  case NormLossParameter_NormType_Lp:
  {
	  Dtype* buf1 = buffer1_.mutable_cpu_data();
	  Dtype* buf2 = buffer2_.mutable_cpu_data();
	  caffe_abs(count, bottom[0]->cpu_data(), buf2 );
	  caffe_powx(count, buf2, p_, buf1);
	  loss = caffe_cpu_asum(count, buf1);
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
Dtype NormLossLayer<Dtype>::normalize_denominator( int count, int batch_size ) {
	  Dtype normalization_denominator = Dtype(1.0);
	  switch ( this->layer_param_.loss_param().normalization() ) {
	  case LossParameter_NormalizationMode_FULL:
		  normalization_denominator = Dtype(count);
		  break;
	  case LossParameter_NormalizationMode_VALID: // same a batch size in this layer
	  case LossParameter_NormalizationMode_BATCH_SIZE:
		  normalization_denominator = batch_size;
		  break;
	  case LossParameter_NormalizationMode_NONE:
		  break;
	  default:
		  LOG(ERROR) << "Unknown NormalizationMode";
	  }
	  return normalization_denominator;
}

template <typename Dtype>
void NormLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	int count = bottom[0]->count();
	const Dtype alpha = top[0]->cpu_diff()[0] /
			this->normalize_denominator( count, bottom[0]->num() );
    if (propagate_down[0]) {
	  switch (norm_type_) {
	  case NormLossParameter_NormType_L1:
	  {
		  Dtype* buf1 = buffer1_.mutable_cpu_data();
		  caffe_cpu_sign(count, bottom[0]->cpu_data(), buf1 );
	      caffe_cpu_axpby(
	          count,              // count
	          alpha,                           // alpha
			  buf1,                       // a
	          Dtype(0),                        // beta
	          bottom[0]->mutable_cpu_diff());  // b
		  break;
	  }
	  case NormLossParameter_NormType_L2:
	  {
	      caffe_cpu_axpby(
	          count,                           // count
	          alpha,                           // alpha
			  bottom[0]->cpu_data(),           // a
	          Dtype(0),                        // beta
	          bottom[0]->mutable_cpu_diff());  // b
		  break;
	  }
	  case NormLossParameter_NormType_Lp:
	  {
		  Dtype* buf1 = buffer1_.mutable_cpu_data();
		  Dtype* buf2 = buffer1_.mutable_cpu_data();
		  caffe_cpu_sign(count, buf2, buf2 );	// avoid div by 0
		  caffe_add(count, bottom[0]->cpu_data(), buf2, buf2);
		  caffe_div(count, buf1, buf2, buf1);
	      caffe_cpu_axpby(
	          count,                           // count
	          alpha,                           // alpha
			  buf1,           			       // a
	          Dtype(0),                        // beta
	          bottom[0]->mutable_cpu_diff());  // b
		  break;
	  }
	  default:
		  LOG(ERROR) << "NormLossLayer: Backward: Unknown norm_type_ : " << norm_type_;
	  }
    }
}

#ifdef CPU_ONLY
STUB_GPU(NormLossLayer);
#endif

INSTANTIATE_CLASS(NormLossLayer);
REGISTER_LAYER_CLASS(NormLoss);

}  // namespace caffe
