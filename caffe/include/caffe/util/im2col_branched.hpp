#ifndef _CAFFE_UTIL_IM2COL_BRANCHED_HPP_
#define _CAFFE_UTIL_IM2COL_BRANCHED_HPP_

namespace caffe_branched {

template <typename Dtype>
void im2col_nd_cpu(const Dtype* data_im, const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    Dtype* data_col);

template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, Dtype* data_col, const int conv_out_factor = 1 );

template <typename Dtype>
void col2im_nd_cpu(const Dtype* data_col, const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    Dtype* data_im);

template <typename Dtype>
void col2im_cpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, Dtype* data_im, const int conv_out_factor = 1 );

template <typename Dtype>
void im2col_nd_gpu(const Dtype* data_im, const int num_spatial_axes,
    const int col_size, const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    Dtype* data_col);

template <typename Dtype>
void im2col_gpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, Dtype* data_col, const int conv_out_factor = 1 );

template <typename Dtype>
void col2im_nd_gpu(const Dtype* data_col, const int num_spatial_axes,
    const int im_size, const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    Dtype* data_im);

template <typename Dtype>
void col2im_gpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, Dtype* data_im, const int conv_out_factor = 1 );

// ----- col <--> vol ===========================================================================================

template <typename Dtype>
void vol2col_cpu(const Dtype* data_vol, const int num_spatial_axes,
    const int* vol_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    Dtype* data_col, const int sample_num = 1);

template <typename Dtype>
void col2vol_cpu(const Dtype* data_col, const int num_spatial_axes,
    const int* vol_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    Dtype* data_vol, const int sample_num = 1);

template <typename Dtype>
void vol2col_gpu(const Dtype* data_vol, const int num_spatial_axes,
    const int num_kernels, const int* vol_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    Dtype* data_col, const int sample_num = 1);

template <typename Dtype>
void col2vol_gpu(const Dtype* data_col, const int num_spatial_axes,
    const int vol_size, const int* vol_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    Dtype* data_vol, const int sample_num = 1);

// vol2col caller

#define vol2col_gpu_caller_MAX_AXES 10

#define vol2col_gpu_caller_ARGLIST_T const int n, const Dtype* data_vol, \
	const int* vol_shape, const int* col_shape, \
	const int* kernel_shape, const int* pad, const int* stride, \
	Dtype* data_col, const int conv_out_factor
#define vol2col_gpu_caller_ARGLIST n, data_vol, vol_shape, col_shape, \
	kernel_shape, pad, stride, data_col, conv_out_factor

template<typename Dtype, int num_axes>
struct vol2col_gpu_kernel_wrapper {
	void operator () (vol2col_gpu_caller_ARGLIST_T) const;
};



// col2vol caller

#define col2vol_gpu_caller_MAX_AXES 10

#define col2vol_gpu_caller_ARGLIST_T const int n, const Dtype* data_col, \
	const int* vol_shape, const int* col_shape, \
	const int* kernel_shape, const int* pad, const int* stride, \
	Dtype* data_vol, const int conv_out_factor
#define col2vol_gpu_caller_ARGLIST n, data_col, vol_shape, col_shape, \
	kernel_shape, pad, stride, data_vol, conv_out_factor

template<typename Dtype, int num_axes>
struct col2vol_gpu_kernel_wrapper {
	void operator () (col2vol_gpu_caller_ARGLIST_T) const;
};

}  // namespace caffe

#endif  // CAFFE_UTIL_IM2COL_HPP_
