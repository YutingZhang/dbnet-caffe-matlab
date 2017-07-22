#if 0

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void DeconvMaskLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
  ConvolutionParameter convolution_param = this->layer_param_.convolution_param();
  CHECK(!convolution_param.has_kernel_size() !=
      !(convolution_param.has_kernel_h() && convolution_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
  CHECK(convolution_param.has_kernel_size() ||
      (convolution_param.has_kernel_h() && convolution_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  CHECK((!convolution_param.has_pad() && convolution_param.has_pad_h()
      && convolution_param.has_pad_w())
      || (!convolution_param.has_pad_h() && !convolution_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!convolution_param.has_stride() && convolution_param.has_stride_h()
      && convolution_param.has_stride_w())
      || (!convolution_param.has_stride_h() && !convolution_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  if (convolution_param.has_kernel_size()) {
    kernel_h_ = kernel_w_ = convolution_param.kernel_size();
  } else {
    kernel_h_ = convolution_param.kernel_h();
    kernel_w_ = convolution_param.kernel_w();
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!convolution_param.has_pad_h()) {
    pad_h_ = pad_w_ = convolution_param.pad();
  } else {
    pad_h_ = convolution_param.pad_h();
    pad_w_ = convolution_param.pad_w();
  }
  if (!convolution_param.has_stride_h()) {
    stride_h_ = stride_w_ = convolution_param.stride();
  } else {
    stride_h_ = convolution_param.stride_h();
    stride_w_ = convolution_param.stride_w();
  }
  channels_ = convolution_param.num_output();
}

template <typename Dtype>
void DeconvMaskLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

	height_ = bottom[0]->height();
	width_ = bottom[0]->width();
	num_ = bottom[0]->num();
	height_out_ = static_cast<int>(static_cast<float>(height_ - 1) * stride_h_ +  kernel_h_ - 2 * pad_h_);
	width_out_ = static_cast<int>(static_cast<float>(width_ - 1) * stride_w_ + kernel_w_ - 2 * pad_w_);

	top[0]->Reshape(num_, channels_, height_out_, width_out_);
	reweight_vec_.Reshape(1, height_out_ + width_out_, 1, 1);

	int map_count = height_out_ * width_out_;
	int channel_count = num_ * channels_;

	Dtype* rwx = reweight_vec_.mutable_cpu_data();
	Dtype* rwy = rwx + width_out_;
	Dtype* rwm = top[0]->mutable_cpu_data();

	for ( int i=0; i<width_out_; ++i ) {
		int pw = DeconvMaskLayer_Utils::compute_reweight(
				i, width_out_, kernel_w_, stride_w_, pad_w_ );
		if (pw)
			rwx[i] = Dtype(1.) / Dtype( pw );
		else
			rwx[i] = 0;
	}
	for ( int i=0; i<height_out_; ++i ) {
		int pw = DeconvMaskLayer_Utils::compute_reweight(
				i, height_out_, kernel_h_, stride_h_, pad_h_ );
		if (pw)
			rwy[i] = Dtype(1.) / Dtype( pw );
		else
			rwy[i] = 0;
	}
	caffe_cpu_gemm<Dtype>( CblasNoTrans, CblasNoTrans, height_out_, width_out_, 1,
			(Dtype) 1., rwy, rwx, (Dtype) 0., rwm );
	for ( int k=1; k<channel_count; ++k ) {	// start from 1
		Dtype* data = rwm + k*map_count;
		caffe_copy( map_count, rwm, data );
	}

}

int DeconvMaskLayer_Utils::compute_reweight( int x, int c, int k, int s, int p ) {
	using std::floor;
	using std::ceil;
	using std::max;
	using std::min;
	float m = floor(float(c+2*p-k)/float(s)+1.f);
	float l = ceil(float(x-k+1+p)/float(s));
	float r = floor(float(x+p)/float(s));
	float n = max(0.f,min(r,m-1.f)-max(l,0.f)+1.f);
	return int(n);
}


#ifdef CPU_ONLY
STUB_GPU(DeconvMaskLayer);
#endif

INSTANTIATE_CLASS(DeconvMaskLayer);
REGISTER_LAYER_CLASS(DeconvMask);

}  // namespace caffe

#endif

