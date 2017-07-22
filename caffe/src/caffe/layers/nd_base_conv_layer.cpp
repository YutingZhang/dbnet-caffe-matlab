#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/nd_conv_layer.hpp"

#include <boost/lexical_cast.hpp>
#include "caffe/util/io.hpp"

namespace caffe_branched_layers {

using namespace caffe;
using namespace caffe_branched;

template <typename Dtype>
ndBaseConvolutionLayer<Dtype>::conv_out_buffer_atomic::conv_out_buffer_atomic(
        ndBaseConvolutionLayer<Dtype>& ndbc) : ndbc_(ndbc), lock_(ndbc.conv_out_buffer_mutex_) {
	ndbc_.conv_out_buffer_src_ = NULL;
}

template <typename Dtype>
ndBaseConvolutionLayer<Dtype>::conv_out_buffer_atomic::~conv_out_buffer_atomic() {
	ndbc_.conv_out_buffer_src_ = NULL;
}

template <typename Dtype>
const Dtype* ndBaseConvolutionLayer<Dtype>::setup_conv_out_buffer_cpu( const Dtype* output ) {
  const Dtype* output1 = output;
  if (parallel_batch_size_>1) {
	  boost::mutex::scoped_lock lock(conv_out_buffer_mutex_,boost::try_to_lock);
	  if ( !lock && conv_out_buffer_src_ == output )
		  return conv_out_buffer_.cpu_data();
	  conv_out_buffer_src_ = output;
      
	  const Dtype* output2=output1;
	  Dtype* output1m=conv_out_buffer_.mutable_cpu_data(); output1 = output1m;
	  int rd[3] = {parallel_batch_size_,conv_out_channels_,conv_out_spatial_dim_};
	  zcaffe_cpu_reordering<Dtype,zcaffe_reorder::D3S01>( rd, output2, output1m );
  }
  return output1;
}


template <typename Dtype>
void ndBaseConvolutionLayer<Dtype>::conv_vol2col_cpu(const Dtype* data, Dtype* col_buff) {
	  vol2col_cpu(data, num_spatial_axes_, conv_input_shape_.cpu_data(),
			  col_buffer_shape_.cpu_data(), kernel_shape_.cpu_data(),
			  pad_.cpu_data(), stride_.cpu_data(), col_buff, parallel_batch_size_ );
}
template <typename Dtype>
void ndBaseConvolutionLayer<Dtype>::conv_col2vol_cpu(const Dtype* col_buff, Dtype* data) {
	  col2vol_cpu(col_buff, num_spatial_axes_, conv_input_shape_.cpu_data(),
			  col_buffer_shape_.cpu_data(), kernel_shape_.cpu_data(),
			  pad_.cpu_data(), stride_.cpu_data(), data, parallel_batch_size_);
}
#ifndef CPU_ONLY

template <typename Dtype>
const Dtype* ndBaseConvolutionLayer<Dtype>::setup_conv_out_buffer_gpu( const Dtype* output ){
  const Dtype* output1 = output;
  if (parallel_batch_size_>1) {
	  boost::mutex::scoped_lock lock(conv_out_buffer_mutex_,boost::try_to_lock);
	  if ( !lock && conv_out_buffer_src_ == output )
		  return conv_out_buffer_.gpu_data();
	  conv_out_buffer_src_ = output;

	  const Dtype* output2=output1;
	  Dtype* output1m=conv_out_buffer_.mutable_gpu_data(); output1 = output1m;
	  int rd[3] = {parallel_batch_size_,conv_out_channels_,conv_out_spatial_dim_};
	  zcaffe_gpu_reordering<Dtype,zcaffe_reorder::D3S01>( rd, output2, output1m );
  }
  return output1;
}


template <typename Dtype>
void ndBaseConvolutionLayer<Dtype>::conv_vol2col_gpu(const Dtype* data, Dtype* col_buff) {
	  vol2col_gpu(data, num_spatial_axes_, num_kernels_vol2col_,
		  conv_input_shape_.gpu_data(), col_buffer_shape_.gpu_data(),
		  kernel_shape_.gpu_data(), pad_.gpu_data(),
		  stride_.gpu_data(), col_buff, parallel_batch_size_);
}
template <typename Dtype>
void ndBaseConvolutionLayer<Dtype>::conv_col2vol_gpu(const Dtype* col_buff, Dtype* data) {
	  col2vol_gpu(col_buff, num_spatial_axes_, num_kernels_col2vol_,
		  conv_input_shape_.gpu_data(), col_buffer_shape_.gpu_data(),
		  kernel_shape_.gpu_data(), pad_.gpu_data(), stride_.gpu_data(), data, parallel_batch_size_);
}
#endif


template <typename Dtype>
void ndBaseConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Configure the kernel size, padding, stride, and inputs.
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
  const int first_spatial_axis = channel_axis_ + 1;
  const int num_axes = bottom[0]->num_axes();
  num_spatial_axes_ = num_axes - first_spatial_axis;
  CHECK_GE(num_spatial_axes_, 1);
  // Setup input dimensions (input_shape_).
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  input_shape_.Reshape(bottom_dim_blob_shape);
  int* input_shape_data = input_shape_.mutable_cpu_data();
  for (int i = 0; i < num_spatial_axes_ + 1; ++i) {
    input_shape_data[i] = bottom[0]->shape(channel_axis_ + i);
  }
  vector<int> spatial_dim_blob_shape(1, num_spatial_axes_);
  // Setup filter kernel dimensions (kernel_shape_).
  kernel_shape_.Reshape(spatial_dim_blob_shape);
  int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
  if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "kernel_h & kernel_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.kernel_size_size())
        << "Either kernel_size or kernel_h/w should be specified; not both.";
    kernel_shape_data[0] = conv_param.kernel_h();
    kernel_shape_data[1] = conv_param.kernel_w();
  } else {
    const int num_kernel_dims = conv_param.kernel_size_size();
    CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
        << "kernel_size must be specified once, or once per spatial dimension "
        << "(kernel_size specified " << num_kernel_dims << " times; "
        << num_spatial_axes_ << " spatial dims);";
      for (int i = 0; i < num_spatial_axes_; ++i) {
        kernel_shape_data[i] =
            conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
      }
  }
  for (int i = 0; i < num_spatial_axes_; ++i) {
    CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
  }
  // Setup stride dimensions (stride_).
  stride_.Reshape(spatial_dim_blob_shape);
  int* stride_data = stride_.mutable_cpu_data();
  if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "stride_h & stride_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.stride_size())
        << "Either stride or stride_h/w should be specified; not both.";
    stride_data[0] = conv_param.stride_h();
    stride_data[1] = conv_param.stride_w();
  } else {
    const int num_stride_dims = conv_param.stride_size();
    CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
          num_stride_dims == num_spatial_axes_)
        << "stride must be specified once, or once per spatial dimension "
        << "(stride specified " << num_stride_dims << " times; "
        << num_spatial_axes_ << " spatial dims);";
    const int kDefaultStride = 1;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
          conv_param.stride((num_stride_dims == 1) ? 0 : i);
      CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
    }
  }
  // Setup pad dimensions (pad_).
  pad_.Reshape(spatial_dim_blob_shape);
  int* pad_data = pad_.mutable_cpu_data();
  if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "pad_h & pad_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.pad_size())
        << "Either pad or pad_h/w should be specified; not both.";
    pad_data[0] = conv_param.pad_h();
    pad_data[1] = conv_param.pad_w();
  } else {
    const int num_pad_dims = conv_param.pad_size();
    CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
          num_pad_dims == num_spatial_axes_)
        << "pad must be specified once, or once per spatial dimension "
        << "(pad specified " << num_pad_dims << " times; "
        << num_spatial_axes_ << " spatial dims);";
    const int kDefaultPad = 0;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
          conv_param.pad((num_pad_dims == 1) ? 0 : i);
    }
  }
  // Special case: vol2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  is_1x1_ = true;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    is_1x1_ &=
        kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;
    if (!is_1x1_) { break; }
  }
  // Configure output channels and groups.
  channels_ = bottom[0]->shape(channel_axis_);
  if ( channel_axis_>0 )
	  closest_num_ = bottom[0]->shape(channel_axis_-1);
  else
	  closest_num_ = 1;

  parallel_batch_size_ = conv_param.parallel_batch_size();
  if ( parallel_batch_size_<1 ) {
	  parallel_batch_size_ = 1;
	  if ( channels_ == 1 && conv_param.single_channel_parallel() )
		  parallel_batch_size_ = closest_num_;
  }
  if (is_1x1_) { parallel_batch_size_ = 1; }

  num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(num_output_, 0);
  group_ = this->layer_param_.convolution_param().group();
  CHECK_EQ(channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";
  if (reverse_dimensions()) {
    conv_out_channels_ = channels_;
    conv_in_channels_ = num_output_;
  } else {
    conv_out_channels_ = num_output_;
    conv_in_channels_ = channels_;
  }
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  bias_term_ = this->layer_param_.convolution_param().bias_term();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    vector<int> weight_shape(2);
    weight_shape[0] = conv_out_channels_;
    weight_shape[1] = conv_in_channels_ / group_;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      weight_shape.push_back(kernel_shape_data[i]);
    }
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    if (bias_term_) {
      vector<int> bias_shape(1, num_output_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void ndBaseConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
  const int first_spatial_axis = channel_axis_ + 1;
  const int num_axes = bottom[0]->num_axes();
  num_spatial_axes_ = num_axes - first_spatial_axis;
  CHECK_GE(num_spatial_axes_, 1);
  num_ = bottom[0]->count(0, channel_axis_);
  CHECK_EQ(bottom[0]->shape(channel_axis_), channels_)
      << "Input size incompatible with convolution kernel.";
  CHECK(!(num_%parallel_batch_size_)) << "num of samples is not divisible by the sample parallel batch size";
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK(bottom[0]->shape() == bottom[bottom_id]->shape())
        << "All inputs must have the same shape.";
  }
  // Shape the tops.
  compute_output_shape();
  vector<int> top_shape = bottom[0]->shape();
  top_shape[channel_axis_] = num_output_;
  top_shape.resize(first_spatial_axis);  // Discard input spatial axes.
  for (int i = 0; i < num_spatial_axes_; ++i) {
    top_shape.push_back(output_shape_[i]);
  }
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }
  if (reverse_dimensions()) {
    conv_out_spatial_dim_ = bottom[0]->count(first_spatial_axis);
  } else {
    conv_out_spatial_dim_ = top[0]->count(first_spatial_axis);
  }
  const int* kernel_shape_data = kernel_shape_.cpu_data();
  kernel_dim_ = conv_in_channels_;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    kernel_dim_ *= kernel_shape_data[i];
  }
  weight_offset_ = conv_out_channels_ * kernel_dim_ / group_ / group_;
  col_offset_ = kernel_dim_ * conv_out_spatial_dim_ / group_;
  output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
  // Setup input dimensions (conv_input_shape_).
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  conv_input_shape_.Reshape(bottom_dim_blob_shape);
  int* conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
  for (int i = 0; i < num_spatial_axes_ + 1; ++i) {
    if (reverse_dimensions()) {
      conv_input_shape_data[i] = top[0]->shape(channel_axis_ + i);
    } else {
      conv_input_shape_data[i] = bottom[0]->shape(channel_axis_ + i);
    }
  }
  // The vol2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.
  col_buffer_shape_par_.clear();
  col_buffer_shape_par_.push_back(kernel_dim_);
  col_buffer_shape_par_.push_back(parallel_batch_size_);
  const int* input_shape_data = input_shape_.cpu_data() + 1;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    if (reverse_dimensions()) {
    	col_buffer_shape_par_.push_back(input_shape_data[i]);
    } else {
    	col_buffer_shape_par_.push_back(output_shape_[i]);
    }
  }
  {
	  col_buffer_shape_.Reshape( vector<int>( 1, col_buffer_shape_par_.size()-1 ) );
	  int* col_buffer_shape_data = col_buffer_shape_.mutable_cpu_data();
      col_buffer_shape_data[0] = col_buffer_shape_par_[0];
	  for (int i = 2; i < col_buffer_shape_par_.size(); ++i ) {
		  col_buffer_shape_data[i-1] = col_buffer_shape_par_[i];
	  }
  }
  col_buffer_.Reshape(vector<int>( col_buffer_shape_par_.begin(), col_buffer_shape_par_.end() ));

  bottom_dim_ = bottom[0]->count(channel_axis_);
  top_dim_ = top[0]->count(channel_axis_);
  num_kernels_vol2col_ = conv_in_channels_ * conv_out_spatial_dim_;
  num_kernels_col2vol_ = reverse_dimensions() ? top_dim_ : bottom_dim_;
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  out_spatial_dim_ = top[0]->count(first_spatial_axis);
  if (bias_term_) {
    vector<int> bias_multiplier_shape(1, out_spatial_dim_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(bias_multiplier_.count(), Dtype(1),
        bias_multiplier_.mutable_cpu_data());
  }

  if (parallel_batch_size_>1) {
	  std::vector<int> conv_out_buffer_shape_(2);
	  conv_out_buffer_shape_[0] = conv_out_channels_;
	  conv_out_buffer_shape_[1] = conv_out_spatial_dim_ * parallel_batch_size_;
	  conv_out_buffer_.Reshape( conv_out_buffer_shape_ );
  } else {
	  std::vector<int> conv_out_buffer_shape_(1,0);
	  conv_out_buffer_.Reshape( conv_out_buffer_shape_ );
  }
}

template <typename Dtype>
void ndBaseConvolutionLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_vol2col) {
  const Dtype* col_buff = input;
  SyncedMemoryGuard col_buffer_guard;
  if (!is_1x1_) {
	col_buffer_guard = col_buffer_.hold_data();
    if (!skip_vol2col) {
      conv_vol2col_cpu(input, col_buffer_.mutable_cpu_data());
    }
    col_buff = col_buffer_.cpu_data();
  }

  Dtype* output1 = output,* output2 = NULL;
  if (parallel_batch_size_>1) { output2=output1; output1=conv_out_buffer_.mutable_cpu_data(); }
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_ * parallel_batch_size_, kernel_dim_ / group_,
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_*parallel_batch_size_ * g,
        (Dtype)0, output1+output_offset_*parallel_batch_size_ * g );
  }
  if (parallel_batch_size_>1) {
  	  int rd[3] = {conv_out_channels_,parallel_batch_size_,conv_out_spatial_dim_};
  	  zcaffe_cpu_reordering<Dtype,zcaffe_reorder::D3S01>( rd, output1, output2 );
  }

}

template <typename Dtype>
void ndBaseConvolutionLayer<Dtype>::forward_cpu_bias(Dtype* output,
    const Dtype* bias) {
	int out_inc = num_output_*out_spatial_dim_;
	for ( int b = 0; b<parallel_batch_size_; ++b ) {
	  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
		  out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.cpu_data(),
		  (Dtype)1., output+out_inc*b);
	}
}

template <typename Dtype>
void ndBaseConvolutionLayer<Dtype>::backward_cpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff;
  SyncedMemoryGuard col_buffer_guard;
  if (is_1x1_) {
    col_buff = input;
  } else {
	  col_buffer_guard = col_buffer_.hold_data();
	  col_buff = col_buffer_.mutable_cpu_data();
  }

  const Dtype* output1 = setup_conv_out_buffer_cpu( output );

  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_ / group_,
        conv_out_spatial_dim_*parallel_batch_size_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output1 + output_offset_*parallel_batch_size_ * g,
        (Dtype)0., col_buff + col_offset_*parallel_batch_size_ * g);
  }

  if (!is_1x1_) {
    conv_col2vol_cpu(col_buff, input);
  }
}

template <typename Dtype>
void ndBaseConvolutionLayer<Dtype>::weight_cpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  SyncedMemoryGuard col_buffer_guard;
  if (!is_1x1_) {
	col_buffer_guard = col_buffer_.hold_data();
    conv_vol2col_cpu(input, col_buffer_.mutable_cpu_data());
    col_buff = col_buffer_.cpu_data();
  }
  const Dtype* output1 = setup_conv_out_buffer_cpu( output );
 
  for (int g = 0; g < group_; ++g) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_ / group_, conv_out_spatial_dim_*parallel_batch_size_,
        (Dtype)1., output1 + output_offset_*parallel_batch_size_ * g, 
        col_buff + col_offset_*parallel_batch_size_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void ndBaseConvolutionLayer<Dtype>::backward_cpu_bias(Dtype* bias,
    const Dtype* input) {
  int out_inc = num_output_*out_spatial_dim_;
  for (int b=0; b<parallel_batch_size_ ; ++b) {
	  caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
		  input+out_inc*b, bias_multiplier_.cpu_data(), 1., bias);
  }
}

#ifndef CPU_ONLY

template <typename Dtype>
void ndBaseConvolutionLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_vol2col) {
  const Dtype* col_buff = input;
  SyncedMemoryGuard col_buffer_guard;
  if (!is_1x1_) {
	col_buffer_guard = col_buffer_.hold_data();
    if (!skip_vol2col) {
      conv_vol2col_gpu(input, col_buffer_.mutable_gpu_data());
    }
    /*
  {
      static int i=0;
      ArrayToGZ( string("./col_buf_") + boost::lexical_cast<std::string>(i) + ".blob.gz", 
              col_buffer_.shape(), col_buffer_.cpu_data() );
      ++i;
      if (i>4)
          LOG(FATAL) << "debug stop";
  }*/
    col_buff = col_buffer_.gpu_data();
  }
  Dtype* output1 = output,* output2 = NULL;
  if (parallel_batch_size_>1) { output2=output1; output1=conv_out_buffer_.mutable_gpu_data(); }
  for (int g = 0; g < group_; ++g) {
	    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
	        group_, conv_out_spatial_dim_*parallel_batch_size_, kernel_dim_ / group_,
	        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * parallel_batch_size_ * g,
	        (Dtype)0., output1 + output_offset_*parallel_batch_size_ * g );
  }
  if (parallel_batch_size_>1) {
      int rd[3] = {conv_out_channels_,parallel_batch_size_,conv_out_spatial_dim_};
      zcaffe_gpu_reordering<Dtype,zcaffe_reorder::D3S01>( rd, output1, output2 );
  }

}

template <typename Dtype>
void ndBaseConvolutionLayer<Dtype>::forward_gpu_bias(Dtype* output,
    const Dtype* bias) {
  int out_inc = num_output_*out_spatial_dim_;
  for ( int b = 0; b<parallel_batch_size_; ++b ) {
	  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
		  out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.gpu_data(),
		  (Dtype)1., output+out_inc*b);
  }
}

template <typename Dtype>
void ndBaseConvolutionLayer<Dtype>::backward_gpu_gemm(const Dtype* output,
    const Dtype* weights, Dtype* input) {
  Dtype* col_buff;
  SyncedMemoryGuard col_buffer_guard;
  if (is_1x1_) {
    col_buff = input;
  } else {
	  col_buffer_guard = col_buffer_.hold_data();
	  col_buff = col_buffer_.mutable_gpu_data();
  }

  const Dtype* output1 = setup_conv_out_buffer_gpu( output );
  
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_ / group_,
        conv_out_spatial_dim_*parallel_batch_size_, conv_out_channels_ / group_,
        (Dtype)1., weights + weight_offset_ * g, output1 + output_offset_*parallel_batch_size_ * g,
        (Dtype)0., col_buff + col_offset_*parallel_batch_size_ * g);
  }

  if (!is_1x1_) {
    conv_col2vol_gpu(col_buff, input);
  }
}

template <typename Dtype>
void ndBaseConvolutionLayer<Dtype>::weight_gpu_gemm(const Dtype* input,
    const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  SyncedMemoryGuard col_buffer_guard;
  if (!is_1x1_) {
	col_buffer_guard = col_buffer_.hold_data();
    conv_vol2col_gpu(input, col_buffer_.mutable_gpu_data());
    col_buff = col_buffer_.gpu_data();
  }
  const Dtype* output1 = setup_conv_out_buffer_gpu( output );
 
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_ / group_,
        kernel_dim_ / group_, conv_out_spatial_dim_*parallel_batch_size_,
        (Dtype)1., output1 + output_offset_*parallel_batch_size_ * g, 
        col_buff + col_offset_*parallel_batch_size_ * g,
        (Dtype)1., weights + weight_offset_ * g);
  }
}

template <typename Dtype>
void ndBaseConvolutionLayer<Dtype>::backward_gpu_bias(Dtype* bias,
    const Dtype* input) {
  int out_inc = num_output_*out_spatial_dim_;
  for (int b=0; b<parallel_batch_size_ ; ++b) {
	  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, out_spatial_dim_, 1.,
		  input+out_inc*b, bias_multiplier_.gpu_data(), 1., bias);
  }
}

#endif  // !CPU_ONLY

INSTANTIATE_CLASS(ndBaseConvolutionLayer);

}  // namespace caffe
