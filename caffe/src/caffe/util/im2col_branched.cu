#include <algorithm>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/im2col_branched.hpp"
#include "caffe/util/zeta/utils.hpp"

#include <boost/preprocessor/repetition/repeat.hpp>

#define INSTANTIATE_VCW_f(_, n, ClassName ) template struct ClassName<float,n+1>;
#define INSTANTIATE_VCW_N_f(n, ClassName ) BOOST_PP_REPEAT(n, INSTANTIATE_VCW_f, ClassName)
#define INSTANTIATE_VCW_d(_, n, ClassName ) template struct ClassName<double,n+1>;
#define INSTANTIATE_VCW_N_d(n, ClassName ) BOOST_PP_REPEAT(n, INSTANTIATE_VCW_d, ClassName)

namespace caffe_branched {

using namespace caffe;

template <typename Dtype>
__global__ void im2col_gpu_kernel(const int n, const Dtype* data_im,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col,
    Dtype* data_col, const int conv_out_factor) {
  CUDA_KERNEL_LOOP(index, n) {
	const int h_index = index / width_col;
	const int h_col = h_index % height_col;
	const int w_col = index % width_col;
	const int c_im = h_index / height_col;
	const int c_col = c_im * kernel_h * kernel_w;
	const int h_offset = h_col * stride_h - pad_h;
	const int w_offset = w_col * stride_w - pad_w;
	const int conv_out_step = conv_out_factor * height_col * width_col;
	Dtype* data_col_ptr = data_col;
	data_col_ptr += c_col * conv_out_step + h_col * width_col + w_col;
    const Dtype* data_im_ptr = data_im;
    data_im_ptr += (c_im * height + h_offset) * width + w_offset;
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        int h_im = h_offset + i;
        int w_im = w_offset + j;
        *data_col_ptr =
            (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
            data_im_ptr[i * width + j] : 0;
        data_col_ptr += conv_out_step;
      }
    }
  }
}

template <typename Dtype>
void im2col_gpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    Dtype* data_col, const int conv_out_factor) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;
  // NOLINT_NEXT_LINE(whitespace/operators)
  im2col_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h,
      pad_w, stride_h, stride_w, height_col, width_col, data_col, conv_out_factor);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void im2col_gpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    float* data_col, const int conv_out_factor);
template void im2col_gpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    double* data_col, const int conv_out_factor);

template <typename Dtype, int num_axes>
__global__ void im2col_nd_gpu_kernel(const int n, const Dtype* data_im,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    Dtype* data_col) {
  int d_temp[num_axes];  // NOLINT(runtime/arrays)
  int d_iter[num_axes];  // NOLINT(runtime/arrays)
  int i;
  CUDA_KERNEL_LOOP(index, n) {
    // Initialize channel_in, computed in the loop below, with intermediate
    // computations used to compute the spatial indices.
    int channel_in = index;
    int channel_out = 1;
    for (i = num_axes - 1; i >= 0; --i) {
      d_temp[i] = channel_in % col_shape[i + 1];
      channel_in /= col_shape[i + 1];
      channel_out *= kernel_shape[i];
    }
    channel_out *= channel_in;
    int data_col_inc = 1;
    for (i = 0; i < num_axes; ++i) {
      channel_out *= col_shape[i + 1];
      channel_out += d_temp[i];
      d_temp[i] = d_temp[i] * stride[i] - pad[i];
      channel_in *= im_shape[i + 1];
      channel_in += d_temp[i];
      data_col_inc *= col_shape[i + 1];
      d_iter[i] = 0;
    }
    Dtype* data_col_ptr = data_col + channel_out;
    const Dtype* data_im_ptr = data_im + channel_in;
    bool incremented;
    do {
      bool in_range = true;
      for (i = 0; i < num_axes; ++i) {
        const int d_iter_im = d_iter[i] + d_temp[i];
        in_range &= d_iter_im >= 0 && d_iter_im < im_shape[i + 1];
        if (!in_range) { break; }
      }
      if (in_range) {
        int data_im_offset = d_iter[0];
        for (i = 1; i < num_axes; ++i) {
          data_im_offset *= im_shape[i + 1];
          data_im_offset += d_iter[i];
        }
        *data_col_ptr = data_im_ptr[data_im_offset];
      } else {
        *data_col_ptr = 0;
      }
      data_col_ptr += data_col_inc;
      incremented = false;
      for (i = num_axes - 1; i >= 0; --i) {
        const int d_max = kernel_shape[i];
        if (d_iter[i] == d_max - 1) {
          d_iter[i] = 0;
        } else {  // d_iter[i] < d_max - 1
          ++d_iter[i];
          incremented = true;
          break;
        }
      }  // for (int i = num_axes - 1; i >= 0; --i)
    } while (incremented);  // do
  }  // CUDA_KERNEL_LOOP(index, n)
}

template <typename Dtype>
void im2col_nd_gpu(const Dtype* data_im, const int num_spatial_axes,
    const int num_kernels, const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    Dtype* data_col) {
  switch (num_spatial_axes) {
  case 1:
    im2col_nd_gpu_kernel<Dtype, 1>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, data_im, im_shape, col_shape,
        kernel_shape, pad, stride, data_col);
    break;
  case 2:
    im2col_nd_gpu_kernel<Dtype, 2>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, data_im, im_shape, col_shape,
        kernel_shape, pad, stride, data_col);
    break;
  case 3:
    im2col_nd_gpu_kernel<Dtype, 3>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, data_im, im_shape, col_shape,
        kernel_shape, pad, stride, data_col);
    break;
  case 4:
    im2col_nd_gpu_kernel<Dtype, 4>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, data_im, im_shape, col_shape,
        kernel_shape, pad, stride, data_col);
    break;
  case 5:
    im2col_nd_gpu_kernel<Dtype, 5>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, data_im, im_shape, col_shape,
        kernel_shape, pad, stride, data_col);
    break;
  case 6:
    im2col_nd_gpu_kernel<Dtype, 6>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, data_im, im_shape, col_shape,
        kernel_shape, pad, stride, data_col);
    break;
  case 7:
    im2col_nd_gpu_kernel<Dtype, 7>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, data_im, im_shape, col_shape,
        kernel_shape, pad, stride, data_col);
    break;
  case 8:
    im2col_nd_gpu_kernel<Dtype, 8>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, data_im, im_shape, col_shape,
        kernel_shape, pad, stride, data_col);
    break;
  case 9:
    im2col_nd_gpu_kernel<Dtype, 9>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, data_im, im_shape, col_shape,
        kernel_shape, pad, stride, data_col);
    break;
  case 10:
    im2col_nd_gpu_kernel<Dtype, 10>  // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>(
        num_kernels, data_im, im_shape, col_shape,
        kernel_shape, pad, stride, data_col);
    break;
  default:
    LOG(FATAL) << "im2col_nd_gpu does not support computation with "
               << num_spatial_axes << " spatial axes";
  }
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void im2col_nd_gpu<float>(const float* data_im,
    const int num_spatial_axes, const int col_size,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    float* data_col);
template void im2col_nd_gpu<double>(const double* data_im,
    const int num_spatial_axes, const int col_size,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    double* data_col);

template <typename Dtype>
__global__ void col2im_gpu_kernel(const int n, const Dtype* data_col,
    const int height, const int width, const int channels,
    const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int height_col, const int width_col,
    Dtype* data_im, const int conv_out_factor) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = 0;
    const int w_im = index % width + pad_w;
    const int h_im = (index / width) % height + pad_h;
    const int c_im = index / (width * height);
    // compute the start and end of the output
    const int w_col_start =
        (w_im < kernel_w) ? 0 : (w_im - kernel_w) / stride_w + 1;
    const int w_col_end =
        min(w_im / stride_w + 1, width_col);
    const int h_col_start =
        (h_im < kernel_h) ? 0 : (h_im - kernel_h) / stride_h + 1;
    const int h_col_end =
        min(h_im / stride_h + 1, height_col);
    const int conv_out_step = conv_out_factor * height_col * width_col;
    /*
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        // the col location: [c * width * height + h_out, w_out]
        int c_col = c_im * kernel_h * kernel_w
            + (h_im - h_col * stride_h) * kernel_w + (w_im - w_col * stride_w);
        // c_col = (c_im * kernel_h * kernel_w + h_im * kernel_w + w_im)
        //      - h_col * stride_h * kernel_w - w_col * stride_w;
        val += data_col[c_col * conv_out_step + h_col * width_col + w_col];
        // ind: offset + ( - h_col * stride_h * kernel_w - w_col * stride_w ) * conv_out_step + h_col * width_col + w_col;
        // ind: offset + h_col * ( - stride_h * kernel_w * conv_out_step + width_col) + w_col * ( 1 - stride_w * conv_out_step );
      }
    }
    */

    // equivalent implementation
    int offset = ( (c_im * kernel_h + h_im) * kernel_w + w_im) * conv_out_step;
    int coeff_h_col = width_col - stride_h * kernel_w * conv_out_step;
    int coeff_w_col = (1 - stride_w * conv_out_step);
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
      }
    }
    data_im[index] = val;

  }
}

template <typename Dtype>
void col2im_gpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, Dtype* data_im, const int conv_out_factor) {
  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  col2im_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_col, height, width, channels, kernel_h, kernel_w,
      pad_h, pad_w, stride_h, stride_w,
      height_col, width_col, data_im, conv_out_factor);
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void col2im_gpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, float* data_im, const int conv_out_factor);
template void col2im_gpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, double* data_im, const int conv_out_factor);

// ----- col <--> vol ===========================================================================================

struct im2col_conv_param {
	int channals;
	int height;
	int width;
	int kernel_h;
	int kernel_w;
	int pad_h;
	int pad_w;
	int stride_h;
	int stride_w;
};

template <typename Dtype, int num_axes>
__global__ void vol2col_gpu_kernel(vol2col_gpu_caller_ARGLIST_T) {
  int d_temp[num_axes];  // NOLINT(runtvole/arrays)
  int d_iter[num_axes];  // NOLINT(runtvole/arrays)
  int i;
  CUDA_KERNEL_LOOP(index, n) {
    // Initialize channel_in, computed in the loop below, with intermediate
    // computations used to compute the spatial indices.
    int channel_in = index;
    int channel_out = 1;
    for (i = num_axes - 1; i >= 0; --i) {
      d_temp[i] = channel_in % col_shape[i + 1];
      channel_in /= col_shape[i + 1];
      channel_out *= kernel_shape[i];	// *** can compute offline
    }
    channel_out *= channel_in;
    int data_col_inc = 1;
    for (i = 0; i < num_axes; ++i) {
      channel_out *= col_shape[i + 1];
      channel_out += d_temp[i];
      d_temp[i] = d_temp[i] * stride[i] - pad[i];
      channel_in *= vol_shape[i + 1];
      channel_in += d_temp[i];
      data_col_inc *= col_shape[i + 1];	// *** can compute offline
      d_iter[i] = 0;
    }
    channel_out  = (channel_out/data_col_inc) * (data_col_inc*conv_out_factor) +
    		(channel_out % data_col_inc);
    data_col_inc *= conv_out_factor;

    Dtype* data_col_ptr = data_col + channel_out;
    const Dtype* data_vol_ptr = data_vol + channel_in;
    bool incremented;
    do {
      bool in_range = true;
      for (i = 0; i < num_axes; ++i) {
        const int d_iter_vol = d_iter[i] + d_temp[i];
        in_range &= d_iter_vol >= 0 && d_iter_vol < vol_shape[i + 1];
        if (!in_range) { break; }
      }
      if (in_range) {
        int data_vol_offset = d_iter[0];
        for (i = 1; i < num_axes; ++i) {
          data_vol_offset *= vol_shape[i + 1];
          data_vol_offset += d_iter[i];
        }
        *data_col_ptr = data_vol_ptr[data_vol_offset];
      } else {
        *data_col_ptr = 0;
      }
      data_col_ptr += data_col_inc;
      incremented = false;
      for (i = num_axes - 1; i >= 0; --i) {
        const int d_max = kernel_shape[i];
        if (d_iter[i] == d_max - 1) {
          d_iter[i] = 0;
        } else {  // d_iter[i] < d_max - 1
          ++d_iter[i];
          incremented = true;
          break;
        }
      }  // for (int i = num_axes - 1; i >= 0; --i)
    } while (incremented);  // do
  }  // CUDA_KERNEL_LOOP(index, n)
}

#define Dtype float
template __global__ void vol2col_gpu_kernel<Dtype,2>(vol2col_gpu_caller_ARGLIST_T);
#undef Dtype
#define Dtype double
template __global__ void vol2col_gpu_kernel<Dtype,2>(vol2col_gpu_caller_ARGLIST_T);
#undef Dtype

template<typename Dtype>
struct vol2col_gpu_kernel_wrapper<Dtype,2> {
	void operator () (vol2col_gpu_caller_ARGLIST_T) const {
		im2col_conv_param p;
		//col_shape is not useful ..
		caffe_copy(3, vol_shape, &(p.channals)); // channels, height, width
		caffe_copy(2, kernel_shape, &(p.kernel_h)); // kernel_h, kernel_w
		caffe_copy(2, pad,    &(p.pad_h));    // pad_h, pad_w
		caffe_copy(2, stride, &(p.stride_h)); // stride_h, stride_w
		im2col_gpu(data_vol, p.channals, p.height, p.width, // channels, height, width
				p.kernel_h, p.kernel_w,	    // kernel_h, kernel_w
				p.pad_h,    p.pad_w,		// pad_h, pad_w
				p.stride_h, p.stride_w,     // stride_h, stride_w
				data_col, conv_out_factor );
	}
};

template<typename Dtype, int num_axes>
void vol2col_gpu_kernel_wrapper<Dtype, num_axes>::operator () (vol2col_gpu_caller_ARGLIST_T) const {
	vol2col_gpu_kernel<Dtype, num_axes>  // NOLINT_NEXT_LINE(whitespace/operators)
		<<<CAFFE_GET_BLOCKS(zeta::get_first_arg(vol2col_gpu_caller_ARGLIST)),
            CAFFE_CUDA_NUM_THREADS>>>( vol2col_gpu_caller_ARGLIST );
		CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_VCW_N_f( vol2col_gpu_caller_MAX_AXES, vol2col_gpu_kernel_wrapper );
INSTANTIATE_VCW_N_d( vol2col_gpu_caller_MAX_AXES, vol2col_gpu_kernel_wrapper );

// --------------------------------------------------------------------------------

template <typename Dtype, int num_axes>
__global__ void col2vol_gpu_kernel( col2vol_gpu_caller_ARGLIST_T ) {
  int d_vol[num_axes];  // NOLINT(runtvole/arrays)
  int d_col_iter[num_axes];  // NOLINT(runtvole/arrays)
  int d_col_start[num_axes];  // NOLINT(runtvole/arrays)
  int d_col_end[num_axes];  // NOLINT(runtvole/arrays)
  CUDA_KERNEL_LOOP(index, n) {
    // Initialize channel_in, computed in the loop below, with intermediate
    // computations used to compute the spatial indices.
    int channel_vol = index;
    // Calculate d_vol (volage dvolensions).
    for (int i = num_axes - 1; i >= 0; --i) {
      d_vol[i] = channel_vol % vol_shape[i + 1] + pad[i];
      channel_vol /= vol_shape[i + 1];
    }
    // Calculate col start/end indices.
    bool done = false;
    int col_inc = conv_out_factor; // ** can compute once offline
    for (int i = 0; i < num_axes; ++i) {
      col_inc*=col_shape[i + 1];
      d_col_start[i] = d_col_iter[i] =
          (d_vol[i] < kernel_shape[i]) ?
          0 : (d_vol[i] - kernel_shape[i]) / stride[i] + 1;
      d_col_end[i] = min(d_vol[i] / stride[i] + 1, col_shape[i + 1]);
      if (d_col_start[i] >= d_col_end[i]) {
        // Skip computation if the dvolension is 0 at any spatial axis --
        // final val will be 0.
        data_vol[index] = 0;
        done = true;
        break;  // for (int i = 0; i < num_axes; ++i)
      }
    }
    if (done) {
      continue;  // CUDA_KERNEL_LOOP(index, n)
    }
    // Loop over the col to compute the output val.
    Dtype val = 0;
    bool incremented = true;
    do {
      // Compute the final offset.
      int fst_offset = 0;
      int kernel_shape_prod = 1;
      for (int i = num_axes - 1; i >= 0; --i) {
        fst_offset +=
            (d_vol[i] - d_col_iter[i] * stride[i]) * kernel_shape_prod;
        kernel_shape_prod *= kernel_shape[i];
      }
      fst_offset += kernel_shape_prod * channel_vol;
      int sec_offset = 0;
      for (int i = 0; i < num_axes; ++i) {
    	  sec_offset *= col_shape[i + 1];
    	  sec_offset += d_col_iter[i];
      }
      val += data_col[fst_offset*col_inc+sec_offset];
      incremented = false;
      for (int i = num_axes - 1; i >= 0; --i) {
        const int d_max = d_col_end[i];
        if (d_col_iter[i] == d_max - 1) {
          d_col_iter[i] = d_col_start[i];
        } else {  // d_col_iter[i] < d_max - 1
          ++d_col_iter[i];
          incremented = true;
          break;  // for (int i = num_axes - 1; i >= 0; --i)
        }
      }  // for (int i = num_axes - 1; i >= 0; --i)
    }  while (incremented);
    data_vol[index] = val;
  }  // CUDA_KERNEL_LOOP(index, n)
}

template<typename Dtype>
struct col2vol_gpu_kernel_wrapper<Dtype,2> {
	void operator () (col2vol_gpu_caller_ARGLIST_T) const {
		im2col_conv_param p;
		//col_shape is not useful ..
		caffe_copy(3, vol_shape, &(p.channals)); // channels, height, width
		caffe_copy(2, kernel_shape, &(p.kernel_h)); // kernel_h, kernel_w
		caffe_copy(2, pad,    &(p.pad_h));    // pad_h, pad_w
		caffe_copy(2, stride, &(p.stride_h)); // stride_h, stride_w
		col2im_gpu(data_col, p.channals, p.height, p.width, // channels, height, width
				p.kernel_h, p.kernel_w,	    // kernel_h, kernel_w
				p.pad_h,    p.pad_w,		// pad_h, pad_w
				p.stride_h, p.stride_w,     // stride_h, stride_w
				data_vol, conv_out_factor );
	}
};

template<typename Dtype, int num_axes>
void col2vol_gpu_kernel_wrapper<Dtype, num_axes>::operator () (col2vol_gpu_caller_ARGLIST_T) const {
		col2vol_gpu_kernel<Dtype, num_axes>  // NOLINT_NEXT_LINE(whitespace/operators)
		<<<CAFFE_GET_BLOCKS(zeta::get_first_arg(col2vol_gpu_caller_ARGLIST)), 
            CAFFE_CUDA_NUM_THREADS>>>( col2vol_gpu_caller_ARGLIST );
		CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_VCW_N_f( col2vol_gpu_caller_MAX_AXES, col2vol_gpu_kernel_wrapper );
INSTANTIATE_VCW_N_d( col2vol_gpu_caller_MAX_AXES, col2vol_gpu_kernel_wrapper );

template <typename Dtype, int num_axes>
__global__ void col2im_nd_gpu_kernel(const int n, const Dtype* data_col,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    Dtype* data_im) {
  int d_im[num_axes];  // NOLINT(runtime/arrays)
  int d_col_iter[num_axes];  // NOLINT(runtime/arrays)
  int d_col_start[num_axes];  // NOLINT(runtime/arrays)
  int d_col_end[num_axes];  // NOLINT(runtime/arrays)
  CUDA_KERNEL_LOOP(index, n) {
    // Initialize channel_in, computed in the loop below, with intermediate
    // computations used to compute the spatial indices.
    int c_im = index;
    // Calculate d_im (image dimensions).
    for (int i = num_axes - 1; i >= 0; --i) {
      d_im[i] = c_im % im_shape[i + 1] + pad[i];
      c_im /= im_shape[i + 1];
    }
    // Calculate col start/end indices.
    bool done = false;
    for (int i = 0; i < num_axes; ++i) {
      d_col_start[i] = d_col_iter[i] =
          (d_im[i] < kernel_shape[i]) ?
          0 : (d_im[i] - kernel_shape[i]) / stride[i] + 1;
      d_col_end[i] = min(d_im[i] / stride[i] + 1, col_shape[i + 1]);
      if (d_col_start[i] >= d_col_end[i]) {
        // Skip computation if the dimension is 0 at any spatial axis --
        // final val will be 0.
        data_im[index] = 0;
        done = true;
        break;  // for (int i = 0; i < num_axes; ++i)
      }
    }
    if (done) {
      continue;  // CUDA_KERNEL_LOOP(index, n)
    }
    // Loop over the col to compute the output val.
    Dtype val = 0;
    bool incremented = true;
    do {
      // Compute the final offset.
      int final_offset = 0;
      int kernel_shape_prod = 1;
      for (int i = num_axes - 1; i >= 0; --i) {
        final_offset +=
            (d_im[i] - d_col_iter[i] * stride[i]) * kernel_shape_prod;
        kernel_shape_prod *= kernel_shape[i];
      }
      final_offset += kernel_shape_prod * c_im;
      for (int i = 0; i < num_axes; ++i) {
        final_offset *= col_shape[i + 1];
        final_offset += d_col_iter[i];
      }
      val += data_col[final_offset];
      incremented = false;
      for (int i = num_axes - 1; i >= 0; --i) {
        const int d_max = d_col_end[i];
        if (d_col_iter[i] == d_max - 1) {
          d_col_iter[i] = d_col_start[i];
        } else {  // d_col_iter[i] < d_max - 1
          ++d_col_iter[i];
          incremented = true;
          break;  // for (int i = num_axes - 1; i >= 0; --i)
        }
      }  // for (int i = num_axes - 1; i >= 0; --i)
    }  while (incremented);
    data_im[index] = val;
  }  // CUDA_KERNEL_LOOP(index, n)
}

template <typename Dtype>
void col2im_nd_gpu(const Dtype* data_col, const int num_spatial_axes,
    const int im_size, const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    Dtype* data_im) {
  switch (num_spatial_axes) {
  case 1:
    col2im_nd_gpu_kernel<Dtype, 1>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
          im_size, data_col, im_shape, col_shape,
          kernel_shape, pad, stride, data_im);
    break;
  case 2:
    col2im_nd_gpu_kernel<Dtype, 2>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
          im_size, data_col, im_shape, col_shape,
          kernel_shape, pad, stride, data_im);
    break;
  case 3:
    col2im_nd_gpu_kernel<Dtype, 3>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
          im_size, data_col, im_shape, col_shape,
          kernel_shape, pad, stride, data_im);
    break;
  case 4:
    col2im_nd_gpu_kernel<Dtype, 4>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
          im_size, data_col, im_shape, col_shape,
          kernel_shape, pad, stride, data_im);
    break;
  case 5:
    col2im_nd_gpu_kernel<Dtype, 5>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
          im_size, data_col, im_shape, col_shape,
          kernel_shape, pad, stride, data_im);
    break;
  case 6:
    col2im_nd_gpu_kernel<Dtype, 6>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
          im_size, data_col, im_shape, col_shape,
          kernel_shape, pad, stride, data_im);
    break;
  case 7:
    col2im_nd_gpu_kernel<Dtype, 7>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
          im_size, data_col, im_shape, col_shape,
          kernel_shape, pad, stride, data_im);
    break;
  case 8:
    col2im_nd_gpu_kernel<Dtype, 8>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
          im_size, data_col, im_shape, col_shape,
          kernel_shape, pad, stride, data_im);
    break;
  case 9:
    col2im_nd_gpu_kernel<Dtype, 9>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
          im_size, data_col, im_shape, col_shape,
          kernel_shape, pad, stride, data_im);
    break;
  case 10:
    col2im_nd_gpu_kernel<Dtype, 10>  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(im_size), CAFFE_CUDA_NUM_THREADS>>>(
          im_size, data_col, im_shape, col_shape,
          kernel_shape, pad, stride, data_im);
    break;
  default:
    LOG(FATAL) << "col2im_nd_gpu does not support computation with "
               << num_spatial_axes << " spatial axes";
  }
  CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void col2im_nd_gpu<float>(const float* data_col,
    const int num_spatial_axes, const int im_size,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    float* data_im);
template void col2im_nd_gpu<double>(const double* data_col,
    const int num_spatial_axes, const int im_size,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    double* data_im);

}  // namespace caffe
