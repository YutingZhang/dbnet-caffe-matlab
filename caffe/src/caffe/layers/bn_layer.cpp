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

#include <algorithm>
#include <vector>

#include "caffe/layers/bn_layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
int BNLayer<Dtype>::MaxTopBlobs() const {
	auto bn_mode = this->layer_param_.bn_param().bn_mode();
	return (
			bn_mode == BNParameter_BNMode_LEARN ||
			bn_mode == BNParameter_BNMode_NORM ||
			bn_mode == BNParameter_BNMode_AUTO_NORM ) ? 3 : 1;
}

template <typename Dtype>
int BNLayer<Dtype>::MaxBottomBlobs() const {
	auto bn_mode = this->layer_param_.bn_param().bn_mode();
	return (
			bn_mode == BNParameter_BNMode_INFERENCE ||
			bn_mode == BNParameter_BNMode_CORRECT ||
			bn_mode == BNParameter_BNMode_AUTO_CORRECT ) ? 3 : 1;
}

template<typename Dtype>
void BNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

	N_ = bottom[0]->count(0,axis_);
	C_ = bottom[0]->shape(axis_);
	G_ = bottom[0]->count(axis_+1);
	cnt_ = N_*C_*G_;

	top[0]->ReshapeLike(*bottom[0]);

	if (top.size() > 1) {
		// top blob for batch mean
		top[1]->Reshape(vector<int>({1, C_}));
	}
	if (top.size() > 2) {
		// top blob for batch variance
		top[2]->Reshape(vector<int>({1, C_}));
	}

	if (bn_mode_ == BNParameter_BNMode_NOTHING) {
		top[0]->ShareDataDiff(*bottom[0]);
		return;
	}

	if (bn_mode_ == BNParameter_BNMode_LEARN) {
		x_norm_.ReshapeLike(*bottom[0]);
		x_norm_.Reshape(vector<int>({1,0}));
	}

	if (bn_mode_ == BNParameter_BNMode_CORRECT || bn_mode_ == BNParameter_BNMode_INFERENCE ||
			bn_mode_ == BNParameter_BNMode_LEARN ) {
		mscale_.Reshape(1, C_, 1, 1);
		mshift_.Reshape(1, C_, 1, 1);
		if (bn_mode_ == BNParameter_BNMode_CORRECT) {
			std_buffer_.Reshape(1, C_, 1, 1);
			mshift_.ShareDiff(*this->blobs_[1]);	// the shift diff is the same
		} else {
			mscale_.ShareDataDiff(*this->blobs_[0]);
			mshift_.ShareDataDiff(*this->blobs_[1]);
		}
	}

	// mean
	spatial_mean_.Reshape(N_, C_, 1, 1);
	batch_mean_.Reshape(1, C_, 1, 1);
	// variance
	spatial_variance_.Reshape(N_, C_, 1, 1);
	batch_variance_.Reshape(1, C_, 1, 1);
	// buffer blob
	buffer_blob_.Reshape(N_, C_, G_, 1);

	// fill spatial multiplier
	spatial_sum_multiplier_.Reshape(1, 1, G_, 1);
	Dtype* spatial_multipl_data = spatial_sum_multiplier_.mutable_cpu_data();
	caffe_set(spatial_sum_multiplier_.count(), Dtype(1), spatial_multipl_data);
	caffe_set(spatial_sum_multiplier_.count(), Dtype(0),
			spatial_sum_multiplier_.mutable_cpu_diff());
	// fill batch multiplier
	batch_sum_multiplier_.Reshape(N_, 1, 1, 1);
	Dtype* batch_multiplier_data = batch_sum_multiplier_.mutable_cpu_data();
	caffe_set(batch_sum_multiplier_.count(), Dtype(1), batch_multiplier_data);
	caffe_set(batch_sum_multiplier_.count(), Dtype(0),
			batch_sum_multiplier_.mutable_cpu_diff());
}
template<typename Dtype>
void BNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

	// Figure out the mode
	bn_mode_ = this->layer_param_.bn_param().bn_mode();
	if (bn_mode_ == BNParameter_BNMode_AUTO_NORM) {
		if (this->phase_ == TRAIN) {
			bn_mode_ = BNParameter_BNMode_NORM;
		} else {
			bn_mode_ = BNParameter_BNMode_NOTHING;
		}
	} else if (bn_mode_ == BNParameter_BNMode_AUTO_CORRECT) {
		if (this->phase_ == TRAIN) {
			bn_mode_ = BNParameter_BNMode_INFERENCE;
		} else {
			bn_mode_ = BNParameter_BNMode_CORRECT;
		}
	}

	if ( bn_mode_ != BNParameter_BNMode_CORRECT && bn_mode_ != BNParameter_BNMode_INFERENCE ) {
		CHECK_EQ( bottom.size(), 1 ) << "Only 1 bottom blob when mode is not CORRECT or INFERENCE";
	}
	if ( bn_mode_ == BNParameter_BNMode_CORRECT ) {
		CHECK( bottom.size() == 1 || bottom.size() == 3 ) << "Bottom blob should be either 1 or 3 when mode is set to CORRECT";
	}
	if ( bn_mode_ != BNParameter_BNMode_NORM && bn_mode_ != BNParameter_BNMode_NOTHING ) {
		CHECK_EQ( top.size(), 1 ) << "Only 1 top blob when mode is not LEARN, NORM or NOTHING";
	}

	// Figure out the dimensions
	axis_ = this->layer_param_.bn_param().axis();
	N_ = bottom[0]->count(0,axis_);
	C_ = bottom[0]->shape(axis_);
	G_ = bottom[0]->count(axis_+1);
	cnt_ = N_*C_*G_;
	var_eps_ = 1e-9;

	// Check if we need to set up the weights
	if (this->blobs_.size() > 0) {
		LOG(INFO) << "Skipping parameter initialization";
	} else {

		if ( bn_mode_ == BNParameter_BNMode_LEARN ||
				bn_mode_ == BNParameter_BNMode_INFERENCE ||
				bn_mode_ == BNParameter_BNMode_CORRECT ) {

			this->blobs_.resize(2);

			// fill scale with scale_filler
			this->blobs_[0].reset(new Blob<Dtype>(1, C_, 1, 1));
			shared_ptr<Filler<Dtype> > scale_filler(
					GetFiller<Dtype>(this->layer_param_.bn_param().scale_filler()));
			scale_filler->Fill(this->blobs_[0].get());

			// fill shift with shift_filler
			this->blobs_[1].reset(new Blob<Dtype>(1, C_, 1, 1));
			shared_ptr<Filler<Dtype> > shift_filler(
					GetFiller<Dtype>(this->layer_param_.bn_param().shift_filler()));
			shift_filler->Fill(this->blobs_[1].get());

		}
	}  // parameter initialization
	this->param_propagate_down_.resize(this->blobs_.size(), true);

}

template <typename Dtype>
template <class DeviceMode>
void BNLayer<Dtype>::Forward_generic(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

	if (bn_mode_ == BNParameter_BNMode_NOTHING) return;

	auto buffer_blob_guard = buffer_blob_.hold_data();

	const Dtype* bottom_data = bottom[0]->data( DeviceMode() );
	Dtype* top_data = top[0]->mutable_data( DeviceMode() );

	const Dtype* const_top_data = top[0]->data( DeviceMode() );
	// forward for normalization
	switch (bn_mode_) {
	case BNParameter_BNMode_LEARN:
	case BNParameter_BNMode_NORM:
	{
		// put the squares of bottom into buffer_blob_
		gcm<DeviceMode>::powx(cnt_, bottom_data, Dtype(2),
				buffer_blob_.mutable_data( DeviceMode() ));

		// computes variance using var(X) = E(X^2) - (EX)^2
		// EX across spatial
		gcm<DeviceMode>::gemv(CblasNoTrans, N_ * C_, G_,
				Dtype(1. / (G_)), bottom_data,
				spatial_sum_multiplier_.data( DeviceMode() ), Dtype(0),
				spatial_mean_.mutable_data( DeviceMode() ));
		// EX across batch
		gcm<DeviceMode>::gemv(CblasTrans, N_, C_, Dtype(1. / N_),
				spatial_mean_.data( DeviceMode() ), batch_sum_multiplier_.data( DeviceMode() ),
				Dtype(0), batch_mean_.mutable_data( DeviceMode() ));

		// E(X^2) across spatial
		gcm<DeviceMode>::gemv(CblasNoTrans, N_ * C_, G_,
				Dtype(1. / (G_)), buffer_blob_.data( DeviceMode() ),
				spatial_sum_multiplier_.data( DeviceMode() ), Dtype(0),
				spatial_variance_.mutable_data( DeviceMode() ));
		// E(X^2) across batch
		gcm<DeviceMode>::gemv(CblasTrans, N_, C_, Dtype(1. / N_),
				spatial_variance_.data( DeviceMode() ), batch_sum_multiplier_.data( DeviceMode() ),
				Dtype(0), batch_variance_.mutable_data( DeviceMode() ));

		gcm<DeviceMode>::powx(batch_mean_.count(), batch_mean_.data( DeviceMode() ), Dtype(2),
				buffer_blob_.mutable_data( DeviceMode() ));  // (EX)^2
		gcm<DeviceMode>::sub(batch_mean_.count(), batch_variance_.data( DeviceMode() ),
				buffer_blob_.data( DeviceMode() ), batch_variance_.mutable_data( DeviceMode() )); // variance

		// save top[1] (batch_mean) and top[2] (batch_variance)
		if (top.size() > 1) {
			caffe_copy(batch_mean_.count(), batch_mean_.data( DeviceMode() ),
					top[1]->mutable_data( DeviceMode() ));
		}
		if (top.size() > 2) {
			caffe_copy(batch_variance_.count(), batch_variance_.data( DeviceMode() ),
					top[2]->mutable_data( DeviceMode() ));
		}

		// do mean and variance normalization
		// subtract mean
		gcm<DeviceMode>::gemm(CblasNoTrans, CblasNoTrans, N_, C_, 1, Dtype(1),
				batch_sum_multiplier_.data( DeviceMode() ), batch_mean_.data( DeviceMode() ),
				Dtype(0), spatial_mean_.mutable_data( DeviceMode() ));

		gcm<DeviceMode>::gemm(CblasNoTrans, CblasNoTrans, N_ * C_, G_, 1,
				Dtype(-1), spatial_mean_.data( DeviceMode() ),
				spatial_sum_multiplier_.data( DeviceMode() ), Dtype(0),
				buffer_blob_.mutable_data( DeviceMode() ));

		gcm<DeviceMode>::add(cnt_, bottom_data, buffer_blob_.data( DeviceMode() ),
				top_data);

		// normalize variance
		gcm<DeviceMode>::add_scalar(batch_variance_.count(), var_eps_,
				batch_variance_.mutable_data( DeviceMode() ));
		gcm<DeviceMode>::powx(batch_variance_.count(), batch_variance_.data( DeviceMode() ),
				Dtype(0.5), batch_variance_.mutable_data( DeviceMode() ));

		gcm<DeviceMode>::gemm(CblasNoTrans, CblasNoTrans, N_, C_, 1, Dtype(1),
				batch_sum_multiplier_.data( DeviceMode() ), batch_variance_.data( DeviceMode() ),
				Dtype(0), spatial_variance_.mutable_data( DeviceMode() ));
		gcm<DeviceMode>::gemm(CblasNoTrans, CblasNoTrans, N_ * C_, G_, 1,
				Dtype(1), spatial_variance_.data( DeviceMode() ),
				spatial_sum_multiplier_.data( DeviceMode() ), Dtype(0),
				buffer_blob_.mutable_data( DeviceMode() ));

		gcm<DeviceMode>::div(cnt_, const_top_data, buffer_blob_.data( DeviceMode() ),
				top_data);
		if ( bn_mode_ == BNParameter_BNMode_LEARN ) {
			// Saving x_norm
			caffe_copy(cnt_, const_top_data,
					x_norm_.mutable_data( DeviceMode() ));
		}
		break;
	}
	case BNParameter_BNMode_CORRECT:
	case BNParameter_BNMode_INFERENCE:
		break;
	default:
		LOG(FATAL) << "Unknown BN mode.";
	}

	// forward for correction
	const Dtype* correction_bottom_data = bottom_data;
	switch (bn_mode_) {
	case BNParameter_BNMode_LEARN:
	{
		correction_bottom_data = top_data;
	}
	case BNParameter_BNMode_CORRECT:
		if (bn_mode_ == BNParameter_BNMode_CORRECT) {
			if (bottom.size()>=3) {
				const Dtype* scale_data = this->blobs_[0]->data( DeviceMode() );
				const Dtype* shift_data = this->blobs_[1]->data( DeviceMode() );
				const Dtype* mean_data  = bottom[1]->data( DeviceMode() );
				const Dtype* var_data   = bottom[2]->data( DeviceMode() );
				Dtype* std_data = std_buffer_.mutable_data( DeviceMode() );
				gcm<DeviceMode>::powx( C_, var_data, Dtype(0.5), std_data );
				gcm<DeviceMode>::add_scalar( C_, Dtype(var_eps_), std_data );
				gcm<DeviceMode>::div( C_, scale_data, std_data, mscale_.mutable_data( DeviceMode() ) );
				Dtype* mshift_data = mshift_.mutable_data( DeviceMode() );
				gcm<DeviceMode>::div( C_, mean_data, std_data, mshift_data );
				gcm<DeviceMode>::mul( C_, scale_data, mshift_data, mshift_data );
				gcm<DeviceMode>::sub( C_, shift_data, mshift_data, mshift_data );
			}
		}
	case BNParameter_BNMode_INFERENCE:
	{
		// scale
		gcm<DeviceMode>::gemm(CblasNoTrans, CblasNoTrans, N_, C_, 1, Dtype(1),
				batch_sum_multiplier_.data( DeviceMode() ), mscale_.data( DeviceMode() ), Dtype(0),
				spatial_variance_.mutable_data( DeviceMode() ));
		gcm<DeviceMode>::gemm(CblasNoTrans, CblasNoTrans, N_ * C_, G_, 1,
				Dtype(1), spatial_variance_.data( DeviceMode() ),
				spatial_sum_multiplier_.data( DeviceMode() ), Dtype(0),
				buffer_blob_.mutable_data( DeviceMode() ));
		gcm<DeviceMode>::mul(cnt_, correction_bottom_data, buffer_blob_.data( DeviceMode() ),
				top_data);

		// shift
		gcm<DeviceMode>::gemm(CblasNoTrans, CblasNoTrans, N_, C_, 1, Dtype(1),
				batch_sum_multiplier_.data( DeviceMode() ), mshift_.data( DeviceMode() ), Dtype(0),
				spatial_mean_.mutable_data( DeviceMode() ));
		gcm<DeviceMode>::gemm(CblasNoTrans, CblasNoTrans, N_ * C_, G_, 1,
				Dtype(1), spatial_mean_.data( DeviceMode() ),
				spatial_sum_multiplier_.data( DeviceMode() ), Dtype(0),
				buffer_blob_.mutable_data( DeviceMode() ));
		gcm<DeviceMode>::add(cnt_, const_top_data, buffer_blob_.data( DeviceMode() ),
				top_data);
		break;
	}
	case BNParameter_BNMode_NORM:
		break;
	default:
		LOG(FATAL) << "Unknown BN mode.";
	}
}

template <typename Dtype>
template <class DeviceMode>
void BNLayer<Dtype>::Backward_generic(const vector<Blob<Dtype>*>& top,
	  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	if (bn_mode_ == BNParameter_BNMode_NOTHING) return;

	auto buffer_blob_guard = buffer_blob_.hold_data();

	const Dtype* top_diff = top[0]->diff( DeviceMode() );
	const Dtype* bottom_data = bottom[0]->data( DeviceMode() );
	Dtype* bottom_diff = bottom[0]->mutable_diff( DeviceMode() );

	// backward for correction
	const Dtype* correction_bottom_data = bottom_data;
	Dtype* correction_bottom_diff = bottom_diff;
	switch (bn_mode_) {
	case BNParameter_BNMode_LEARN:
	{
		correction_bottom_data = x_norm_.data( DeviceMode() );
		correction_bottom_diff = buffer_blob_.mutable_data( DeviceMode() );
	}
	case BNParameter_BNMode_CORRECT:
	case BNParameter_BNMode_INFERENCE:
	{
		const Dtype* scale_data = this->blobs_[0]->data( DeviceMode() );

		// Propagate layer to parameters
		// gradient w.r.t. scale
		gcm<DeviceMode>::mul(cnt_, correction_bottom_data, top_diff,
				buffer_blob_.mutable_data( DeviceMode() ));
		// EX across spatial
		gcm<DeviceMode>::gemv(CblasNoTrans, N_ * C_, G_, Dtype(1),
				buffer_blob_.data( DeviceMode() ), spatial_sum_multiplier_.data( DeviceMode() ),
				Dtype(0), spatial_variance_.mutable_diff( DeviceMode() ));
		// EX across batch
		gcm<DeviceMode>::gemv(CblasTrans, N_, C_, Dtype(1),
				spatial_variance_.diff( DeviceMode() ), batch_sum_multiplier_.data( DeviceMode() ),
				Dtype(0), mscale_.mutable_diff( DeviceMode() ) );

		// gradient w.r.t. shift
		// EX across spatial
		gcm<DeviceMode>::gemv(CblasNoTrans, N_ * C_, G_, Dtype(1),
				top_diff, spatial_sum_multiplier_.data( DeviceMode() ), Dtype(0),
				spatial_mean_.mutable_diff( DeviceMode() ));
		// EX across batch
		gcm<DeviceMode>::gemv(CblasTrans, N_, C_, Dtype(1),
				spatial_mean_.diff( DeviceMode() ), batch_sum_multiplier_.data( DeviceMode() ),
				Dtype(0), mshift_.mutable_diff( DeviceMode() ) );

		// Propagate down
		// put scale * top_diff to buffer_blob_
		gcm<DeviceMode>::gemm(CblasNoTrans, CblasNoTrans, N_, C_, 1, Dtype(1),
				batch_sum_multiplier_.data( DeviceMode() ), scale_data, Dtype(0),
				spatial_variance_.mutable_data( DeviceMode() ));
		gcm<DeviceMode>::gemm(CblasNoTrans, CblasNoTrans, N_ * C_, G_, 1,
				Dtype(1), spatial_variance_.data( DeviceMode() ),
				spatial_sum_multiplier_.data( DeviceMode() ), Dtype(0),
				buffer_blob_.mutable_data( DeviceMode() ));
		gcm<DeviceMode>::mul(cnt_, top_diff, buffer_blob_.data( DeviceMode() ),
				correction_bottom_diff);
	}
	{
		if (bn_mode_==BNParameter_BNMode_CORRECT && bottom.size()>=3) {
			const Dtype* std_data = std_buffer_.data( DeviceMode() );
			Dtype* scale_diff = this->blobs_[0]->mutable_diff( DeviceMode() );
			gcm<DeviceMode>::div( C_, mscale_.diff( DeviceMode() ), std_data, scale_diff );
			// SHIFT diff is shared
			// Dtype* shift_diff = this->blobs_[1]->mutable_diff( DeviceMode() );
		}
		break;
	}
	case BNParameter_BNMode_NORM:
		break;
	default:
		LOG(FATAL) << "Unknown BN mode.";
	}

	// backward for normalization
	const Dtype* normalization_top_data = top[0]->data( DeviceMode() );
	const Dtype* normalization_top_diff = top_diff;
	switch (bn_mode_) {
	case BNParameter_BNMode_LEARN:
		normalization_top_data = x_norm_.data( DeviceMode() );
		buffer_blob_.data( DeviceMode() );
	case BNParameter_BNMode_NORM:
		// use new top diff for computation
		gcm<DeviceMode>::mul(cnt_, normalization_top_data,
				normalization_top_diff, bottom_diff);
		// EX across spatial
		gcm<DeviceMode>::gemv(CblasNoTrans, N_ * C_, G_, Dtype(1),
				bottom_diff, spatial_sum_multiplier_.data( DeviceMode() ), Dtype(0),
				spatial_mean_.mutable_data( DeviceMode() ));
		// EX across batch
		gcm<DeviceMode>::gemv(CblasTrans, N_, C_, Dtype(1),
				spatial_mean_.data( DeviceMode() ), batch_sum_multiplier_.data( DeviceMode() ),
				Dtype(0), batch_mean_.mutable_data( DeviceMode() ));

		gcm<DeviceMode>::gemm(CblasNoTrans, CblasNoTrans, N_, C_, 1, Dtype(1),
				batch_sum_multiplier_.data( DeviceMode() ), batch_mean_.data( DeviceMode() ),
				Dtype(0), spatial_mean_.mutable_data( DeviceMode() ));
		gcm<DeviceMode>::gemm(CblasNoTrans, CblasNoTrans, N_ * C_, G_, 1,
				Dtype(1), spatial_mean_.data( DeviceMode() ),
				spatial_sum_multiplier_.data( DeviceMode() ), Dtype(0), bottom_diff);

		gcm<DeviceMode>::mul(cnt_, normalization_top_data, bottom_diff,
				bottom_diff);

		// EX across spatial
		gcm<DeviceMode>::gemv(CblasNoTrans, N_ * C_, G_, Dtype(1),
				normalization_top_diff, spatial_sum_multiplier_.data( DeviceMode() ),
				Dtype(0), spatial_mean_.mutable_data( DeviceMode() ));
		// EX across batch
		gcm<DeviceMode>::gemv(CblasTrans, N_, C_, Dtype(1),
				spatial_mean_.data( DeviceMode() ), batch_sum_multiplier_.data( DeviceMode() ),
				Dtype(0), batch_mean_.mutable_data( DeviceMode() ));

		gcm<DeviceMode>::gemm(CblasNoTrans, CblasNoTrans, N_, C_, 1, Dtype(1),
				batch_sum_multiplier_.data( DeviceMode() ), batch_mean_.data( DeviceMode() ),
				Dtype(0), spatial_mean_.mutable_data( DeviceMode() ));
		gcm<DeviceMode>::gemm(CblasNoTrans, CblasNoTrans, N_ * C_, G_, 1,
				Dtype(1), spatial_mean_.data( DeviceMode() ),
				spatial_sum_multiplier_.data( DeviceMode() ), Dtype(1), bottom_diff);

		gcm<DeviceMode>::axpby(cnt_, Dtype(1), normalization_top_diff,
				Dtype(-1. / (N_ * G_)), bottom_diff);

		// put the squares of bottom into buffer_blob_
		gcm<DeviceMode>::powx(cnt_, bottom_data, Dtype(2),
				buffer_blob_.mutable_data( DeviceMode() ));

		gcm<DeviceMode>::gemm(CblasNoTrans, CblasNoTrans, N_, C_, 1, Dtype(1),
				batch_sum_multiplier_.data( DeviceMode() ), batch_variance_.data( DeviceMode() ),
				Dtype(0), spatial_variance_.mutable_data( DeviceMode() ));
		gcm<DeviceMode>::gemm(CblasNoTrans, CblasNoTrans, N_ * C_, G_, 1,
				Dtype(1), spatial_variance_.data( DeviceMode() ),
				spatial_sum_multiplier_.data( DeviceMode() ), Dtype(0),
				buffer_blob_.mutable_data( DeviceMode() ));

		gcm<DeviceMode>::div(cnt_, bottom_diff, buffer_blob_.data( DeviceMode() ),
				bottom_diff);
		break;
	case BNParameter_BNMode_CORRECT:
	case BNParameter_BNMode_INFERENCE:
		break;
	default:
		LOG(FATAL) << "Unknown BN mode.";
	}
}


template<typename Dtype>
void BNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	Forward_generic<mCPU>(bottom,top);
}

template<typename Dtype>
void BNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
	Backward_generic<mCPU>(top,propagate_down,bottom);
}

#ifdef CPU_ONLY
template<typename Dtype>
void BNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	NO_GPU;
}

template<typename Dtype>
void BNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
	NO_GPU;
}
#else

template<typename Dtype>
void BNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	Forward_generic<mGPU>(bottom,top);
}

template<typename Dtype>
void BNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
	Backward_generic<mGPU>(top,propagate_down,bottom);
}

#endif

INSTANTIATE_CLASS(BNLayer);
REGISTER_LAYER_CLASS(BN);

}  // namespace caffe
