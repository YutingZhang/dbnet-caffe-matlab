#ifndef CAFFE_BRANCHED_ND_CONV_LAYER_HPP_
#define CAFFE_BRANCHED_ND_CONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/util/im2col_branched.hpp"

namespace caffe_branched_layers {

using namespace caffe;

/**
 * @brief Abstract base class that factors out the BLAS code common to
 *        ndConvolutionLayer and ndDeconvolutionLayer.
 */
template <typename Dtype>
class ndBaseConvolutionLayer : public Layer<Dtype> {
 public:
  explicit ndBaseConvolutionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline bool EqualNumBottomTopBlobs() const { return true; }

 protected:
  // Helper functions that abstract away the column buffer and gemm arguments.
  // The last argument in forward_cpu_gemm is so that we can skip the vol2col if
  // we just called weight_cpu_gemm with the same input.
  void forward_cpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* output, bool skip_vol2col = false);
  void forward_cpu_bias(Dtype* output, const Dtype* bias);
  void backward_cpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* output);
  void weight_cpu_gemm(const Dtype* input, const Dtype* output, Dtype*
      weights);
  void backward_cpu_bias(Dtype* bias, const Dtype* input);

#ifndef CPU_ONLY
  void forward_gpu_gemm(const Dtype* col_input, const Dtype* weights,
      Dtype* output, bool skip_vol2col = false);
  void forward_gpu_bias(Dtype* output, const Dtype* bias);
  void backward_gpu_gemm(const Dtype* input, const Dtype* weights,
      Dtype* col_output);
  void weight_gpu_gemm(const Dtype* col_input, const Dtype* output, Dtype*
      weights);
  void backward_gpu_bias(Dtype* bias, const Dtype* input);
#endif

  // reverse_dimensions should return true iff we are implementing deconv, so
  // that conv helpers know which dimensions are which.
  virtual bool reverse_dimensions() = 0;
  // Compute height_out_ and width_out_ from other parameters.
  virtual void compute_output_shape() = 0;

  /// @brief The spatial dimensions of a filter kernel.
  Blob<int> kernel_shape_;
  /// @brief The spatial dimensions of the stride.
  Blob<int> stride_;
  /// @brief The spatial dimensions of the padding.
  Blob<int> pad_;
  /// @brief The spatial dimensions of the convolution input.
  Blob<int> conv_input_shape_;
  /// @brief The spatial dimensions of the input.
  Blob<int> input_shape_;
  /// @brief The spatial dimensions of the col_buffer.
  vector<int> col_buffer_shape_par_;
  Blob<int>   col_buffer_shape_;
  /// @brief The spatial dimensions of the output.
  vector<int> output_shape_;

  Blob<Dtype>   conv_out_buffer_;
  const Dtype*  conv_out_buffer_src_;
  boost::mutex  conv_out_buffer_mutex_;
  const Dtype* setup_conv_out_buffer_cpu( const Dtype* output );
  const Dtype* setup_conv_out_buffer_gpu( const Dtype* output );

  class conv_out_buffer_atomic {
      ndBaseConvolutionLayer<Dtype>& ndbc_;
	  boost::mutex::scoped_lock lock_;
  public:
	  conv_out_buffer_atomic(ndBaseConvolutionLayer<Dtype>& ndbc);
	  ~conv_out_buffer_atomic();
  };

  friend class conv_out_buffer_atomic;

  int num_spatial_axes_;
  int bottom_dim_;
  int top_dim_;

  int channel_axis_;
  int num_;
  int channels_;
  int group_;
  int num_output_;
  bool bias_term_;
  bool is_1x1_;

  int closest_num_;
  int parallel_batch_size_;

 private:
  // wrap vol2col/col2vol so we don't have to remember the (long) argument lists
  inline void conv_vol2col_cpu(const Dtype* data, Dtype* col_buff);
  inline void conv_col2vol_cpu(const Dtype* col_buff, Dtype* data);
#ifndef CPU_ONLY
  inline void conv_vol2col_gpu(const Dtype* data, Dtype* col_buff);
  inline void conv_col2vol_gpu(const Dtype* col_buff, Dtype* data);
#endif

  int num_kernels_vol2col_;
  int num_kernels_col2vol_;
  int conv_out_channels_;
  int conv_in_channels_;
  int conv_out_spatial_dim_;
  int out_spatial_dim_;
  int kernel_dim_;
  int weight_offset_;
  int col_offset_;
  int output_offset_;

  DynamicBlobAt<Dtype,SyncedMemoryAllocator::CONV_COL> col_buffer_;
  Blob<Dtype> bias_multiplier_;
};

/**
 * @brief Convolves the input image with a bank of learned filters,
 *        and (optionally) adds biases.
 *
 *   Caffe convolves by reduction to matrix multiplication. This achieves
 *   high-throughput and generality of input and filter dimensions but comes at
 *   the cost of memory for matrices. This makes use of efficiency in BLAS.
 *
 *   The input is "vol2col" transformed to a channel K' x H x W data matrix
 *   for multiplication with the N x K' x H x W filter matrix to yield a
 *   N' x H x W output matrix that is then "col2vol" restored. K' is the
 *   input channel * kernel height * kernel width dimension of the unrolled
 *   inputs so that the vol2col matrix has a column for each input region to
 *   be filtered. col2vol restores the output spatial structure by rolling up
 *   the output channel N' columns of the output matrix.
 */
template <typename Dtype>
class ndConvolutionLayer : public ndBaseConvolutionLayer<Dtype> {
 public:
  /**
   * @param param provides ConvolutionParameter convolution_param,
   *    with ndConvolutionLayer options:
   *  - num_output. The number of filters.
   *  - kernel_size / kernel_h / kernel_w. The filter dimensions, given by
   *  kernel_size for square filters or kernel_h and kernel_w for rectangular
   *  filters.
   *  - stride / stride_h / stride_w (\b optional, default 1). The filter
   *  stride, given by stride_size for equal dimensions or stride_h and stride_w
   *  for different strides. By default the convolution is dense with stride 1.
   *  - pad / pad_h / pad_w (\b optional, default 0). The zero-padding for
   *  convolution, given by pad for equal dimensions or pad_h and pad_w for
   *  different padding. Input padding is computed implicitly instead of
   *  actually padding.
   *  - group (\b optional, default 1). The number of filter groups. Group
   *  convolution is a method for reducing parameterization by selectively
   *  connecting input and output channels. The input and output channel dimensions must be divisible
   *  by the number of groups. For group @f$ \geq 1 @f$, the
   *  convolutional filters' input and output channels are separated s.t. each
   *  group takes 1 / group of the input channels and makes 1 / group of the
   *  output channels. Concretely 4 input channels, 8 output channels, and
   *  2 groups separate input channels 1-2 and output channels 1-4 into the
   *  first group and input channels 3-4 and output channels 5-8 into the second
   *  group.
   *  - bias_term (\b optional, default true). Whether to have a bias.
   *  - engine: convolution has CAFFE (matrix multiplication) and CUDNN (library
   *    kernels + stream parallelism) engines.
   */
  explicit ndConvolutionLayer(const LayerParameter& param)
      : ndBaseConvolutionLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "ndConvolution"; }

 protected:

  using typename ndBaseConvolutionLayer<Dtype>::conv_out_buffer_atomic;

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual inline bool reverse_dimensions() { return false; }
  virtual void compute_output_shape();
};

/**
 * @brief Convolve the input with a bank of learned filters, and (optionally)
 *        add biases, treating filters and convolution parameters in the
 *        opposite sense as ndConvolutionLayer.
 *
 *   ndConvolutionLayer computes each output value by dotting an input window with
 *   a filter; ndDeconvolutionLayer multiplies each input value by a filter
 *   elementwise, and sums over the resulting output windows. In other words,
 *   ndDeconvolutionLayer is ndConvolutionLayer with the forward and backward passes
 *   reversed. ndDeconvolutionLayer reuses ConvolutionParameter for its
 *   parameters, but they take the opposite sense as in ndConvolutionLayer (so
 *   padding is removed from the output rather than added to the input, and
 *   stride results in upsampling rather than downsampling).
 */
template <typename Dtype>
class ndDeconvolutionLayer : public ndBaseConvolutionLayer<Dtype> {
 public:
  explicit ndDeconvolutionLayer(const LayerParameter& param)
      : ndBaseConvolutionLayer<Dtype>(param) {}

  virtual inline const char* type() const { return "ndDeconvolution"; }

 protected:

  using typename ndBaseConvolutionLayer<Dtype>::conv_out_buffer_atomic;

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual inline bool reverse_dimensions() { return true; }
  virtual void compute_output_shape();
};

}

namespace caffe {
    using namespace caffe_branched_layers;
}

#endif

