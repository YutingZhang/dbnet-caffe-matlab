#include <vector>

#include "caffe/util/im2col_branched.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/zeta/rec_func_tpl_caller.hpp"

//#include <fstream>

namespace caffe_branched {

using namespace caffe;

template <typename Dtype>
void im2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    Dtype* data_col,const int conv_out_factor) {

  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  int conv_out_step = conv_out_factor * height_col * width_col;
  int channels_col = channels * kernel_h * kernel_w;
  for (int c_col = 0; c_col < channels_col; ++c_col) {
    const int w_offset = c_col % kernel_w;
    const int h_offset = (c_col / kernel_w) % kernel_h;
    const int c_im = c_col / kernel_h / kernel_w;
    for (int h_col = 0; h_col < height_col; ++h_col) {
      for (int w_col = 0; w_col < width_col; ++w_col) {
        int h_im = h_col * stride_h - pad_h + h_offset;
        int w_im = w_col * stride_w - pad_w + w_offset;
        data_col[c_col * conv_out_step + h_col * width_col + w_col] =
            (h_im >= 0 && h_im < height && w_im >= 0 && w_im < width) ?
            data_im[(c_im * height + h_im) * width + w_im] : 0;
      }
    }
  }
}

// Explicit instantiation
template void im2col_cpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, float* data_col, const int conv_out_factor);
template void im2col_cpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, double* data_col, const int conv_out_factor);

template <typename Dtype>
inline void im2col_nd_core_cpu(const Dtype* data_input, const bool im2col,
    const int num_spatial_axes, const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    Dtype* data_output) {
  if (!im2col) {
    int im_size = im_shape[0];
    for (int i = 0; i < num_spatial_axes; ++i) {
      im_size *= im_shape[1 + i];
    }
    caffe_set(im_size, Dtype(0), data_output);
  }
  int kernel_size = 1;
  for (int i = 0; i < num_spatial_axes; ++i) {
    kernel_size *= kernel_shape[i];
  }
  const int channels_col = col_shape[0];
  vector<int> d_offset(num_spatial_axes, 0);
  vector<int> d_iter(num_spatial_axes, 0);
  for (int c_col = 0; c_col < channels_col; ++c_col) {
    // Loop over spatial axes in reverse order to compute a per-axis offset.
    int offset = c_col;
    for (int d_i = num_spatial_axes - 1; d_i >= 0; --d_i) {
      if (d_i < num_spatial_axes - 1) {
        offset /= kernel_shape[d_i + 1];
      }
      d_offset[d_i] = offset % kernel_shape[d_i];
    }
    for (bool incremented = true; incremented; ) {
      // Loop over spatial axes in forward order to compute the indices in the
      // image and column, and whether the index lies in the padding.
      int index_col = c_col;
      int index_im = c_col / kernel_size;
      bool is_padding = false;
      for (int d_i = 0; d_i < num_spatial_axes; ++d_i) {
        const int d = d_iter[d_i];
        const int d_im = d * stride[d_i] - pad[d_i] + d_offset[d_i];
        is_padding |= d_im < 0 || d_im >= im_shape[d_i + 1];
        index_col *= col_shape[d_i + 1];
        index_col += d;
        index_im *= im_shape[d_i + 1];
        index_im += d_im;
      }
      if (im2col) {
        if (is_padding) {
          data_output[index_col] = 0;
        } else {
          data_output[index_col] = data_input[index_im];
        }
      } else if (!is_padding) {  // col2im
        data_output[index_im] += data_input[index_col];
      }
      // Loop over spatial axes in reverse order to choose an index,
      // like counting.
      incremented = false;
      for (int d_i = num_spatial_axes - 1; d_i >= 0; --d_i) {
        const int d_max = col_shape[d_i + 1];
        DCHECK_LT(d_iter[d_i], d_max);
        if (d_iter[d_i] == d_max - 1) {
          d_iter[d_i] = 0;
        } else {  // d_iter[d_i] < d_max - 1
          ++d_iter[d_i];
          incremented = true;
          break;
        }
      }
    }  // while(incremented) {
  }  // for (int c = 0; c < channels_col; ++c) {
}

template <typename Dtype>
void im2col_nd_cpu(const Dtype* data_im, const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    Dtype* data_col) {
  const bool kIm2Col = true;
  im2col_nd_core_cpu(data_im, kIm2Col, num_spatial_axes, im_shape, col_shape,
                  kernel_shape, pad, stride, data_col);
}

// Explicit instantiation
template void im2col_nd_cpu<float>(const float* data_im,
    const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    float* data_col);
template void im2col_nd_cpu<double>(const double* data_im,
    const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    double* data_col);

template <typename Dtype>
void col2im_cpu(const Dtype* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    Dtype* data_im, const int conv_out_factor) {

  caffe_set(height * width * channels, Dtype(0), data_im);
  const int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  const int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  const int conv_out_step = conv_out_factor * height_col * width_col;
  const int channels_col = channels * kernel_h * kernel_w;
  for (int c_col = 0; c_col < channels_col; ++c_col) {
    int w_offset = c_col % kernel_w;
    int h_offset = (c_col / kernel_w) % kernel_h;
    int c_im = c_col / kernel_h / kernel_w;
    for (int h_col = 0; h_col < height_col; ++h_col) {
      for (int w_col = 0; w_col < width_col; ++w_col) {
        int h_im = h_col * stride_h - pad_h + h_offset;
        int w_im = w_col * stride_w - pad_w + w_offset;
        if (h_im >= 0 && h_im < height && w_im >= 0 && w_im < width)
          data_im[(c_im * height + h_im) * width + w_im] +=
              data_col[c_col * conv_out_step + h_col * width_col + w_col];
      }
    }
  }
}

// Explicit instantiation
template void col2im_cpu<float>(const float* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, float* data_im, const int conv_out_factor);
template void col2im_cpu<double>(const double* data_col, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, double* data_im, const int conv_out_factor);

// ----- col <--> vol ===========================================================================================

template <typename Dtype>
inline void vol2col_core_cpu(const Dtype* data_input, const bool vol2col,
    const int num_spatial_axes, const int* vol_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    Dtype* data_output, const int conv_out_factor) {

  if (!vol2col) {
    int vol_size = vol_shape[0];
    for (int i = 0; i < num_spatial_axes; ++i) {
      vol_size *= vol_shape[1 + i];
    }
    caffe_set(vol_size, Dtype(0), data_output);
  }
  int kernel_size = 1;
  int col_inc = conv_out_factor;
  for (int i = 0; i < num_spatial_axes; ++i) {
    kernel_size *= kernel_shape[i];
    col_inc *= col_shape[i + 1];
  }
  const int channels_col = col_shape[0];
  vector<int> d_offset(num_spatial_axes, 0);
  vector<int> d_iter(num_spatial_axes, 0);
  for (int c = 0; c < channels_col; ++c) {
    // Loop over spatial axes in reverse order to compute a per-axis offset.
    int offset = c;
    for (int d_i = num_spatial_axes - 1; d_i >= 0; --d_i) {
      if (d_i < num_spatial_axes - 1) {
        offset /= kernel_shape[d_i + 1];
      }
      d_offset[d_i] = offset % kernel_shape[d_i];
    }
    for (bool incremented = true; incremented; ) {
      // Loop over spatial axes in forward order to compute the indices in the
      // volage and column, and whether the index lies in the padding.
      int index_col = 0;
      int index_vol = c / kernel_size;
      bool is_padding = false;
      for (int d_i = 0; d_i < num_spatial_axes; ++d_i) {
        const int d = d_iter[d_i];
        const int d_pad = d * stride[d_i] - pad[d_i] + d_offset[d_i];
        is_padding |= d_pad < 0 || d_pad >= vol_shape[d_i + 1];
        index_col *= col_shape[d_i + 1];
        index_col += d;
        index_vol *= vol_shape[d_i + 1];
        index_vol += d_pad;
      }
      index_col += c*col_inc;
      if (vol2col) {
        if (is_padding) {
          data_output[index_col] = 0;
        } else {
          data_output[index_col] = data_input[index_vol];
        }
      } else if (!is_padding) {  // col2vol
        data_output[index_vol] += data_input[index_col];
      }
      // Loop over spatial axes in reverse order to choose an index,
      // like counting.
      incremented = false;
      for (int d_i = num_spatial_axes - 1; d_i >= 0; --d_i) {
        const int d_max = col_shape[d_i + 1];
        DCHECK_LT(d_iter[d_i], d_max);
        if (d_iter[d_i] == d_max - 1) {
          d_iter[d_i] = 0;
        } else {  // d_iter[d_i] < d_max - 1
          ++d_iter[d_i];
          incremented = true;
          break;
        }
      }
    }  // while(incremented) {
  }  // for (int c = 0; c < channels_col; ++c) {
}

template <typename Dtype>
void vol2col_cpu_internal(const Dtype* data_vol, const int num_spatial_axes,
    const int* vol_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    Dtype* data_col, const int conv_out_factor) {
	  if ( num_spatial_axes == 2 ) {
		  //col_shape is not useful ..
		im2col_cpu(data_vol,
				vol_shape[0], vol_shape[1], vol_shape[2], // channels, height, width
				kernel_shape[0], kernel_shape[1],	// kernel_h, kernel_w
				pad[0],    pad[1],		// pad_h, pad_w
				stride[0], stride[1],	// stride_h, stride_w
				data_col, conv_out_factor );
	  } else {
		  const bool kVol2Col = true;
		  vol2col_core_cpu(data_vol, kVol2Col, num_spatial_axes, vol_shape, col_shape,
				  kernel_shape, pad, stride, data_col, conv_out_factor);
	  }
}

template <typename Dtype>
void vol2col_cpu(const Dtype* data_vol, const int num_spatial_axes,
    const int* vol_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    Dtype* data_col, const int sample_num ) {
	//
	int col_inc = 1;
	for ( int i=0; i<num_spatial_axes; ++i )
		col_inc *= col_shape[i+1];
	int vol_size = 1;
	for ( int i=0; i<num_spatial_axes+1; ++i )
		vol_size *= vol_shape[i];
    
	for ( int b=0; b<sample_num; ++b ) {
		vol2col_cpu_internal<Dtype>( data_vol+vol_size*b, num_spatial_axes,
			    vol_shape, col_shape, kernel_shape, pad, stride,
			    data_col+col_inc*b, sample_num );
	}
#if 0
    {
        LOG(INFO) << "DEBUG: start";
        std::ofstream out("a.txt");
        for ( int k=0; k<col_inc*sample_num*5; ++k ) {
            out << data_col[k] << std::endl;
        }
        LOG(INFO) << "DEBUG: end";
    }
    LOG(FATAL) << "DEBUG end";
#endif
}

// Explicit instantiation
template void vol2col_cpu<float>(const float* data_vol,
    const int num_spatial_axes,
    const int* vol_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    float* data_col, const int sample_num);
template void vol2col_cpu<double>(const double* data_vol,
    const int num_spatial_axes,
    const int* vol_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    double* data_col, const int sample_num);

template <typename Dtype>
void col2vol_cpu_internal(const Dtype* data_col, const int num_spatial_axes,
    const int* vol_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    Dtype* data_vol, const int conv_out_factor) {
	  if ( num_spatial_axes == 2 ) {
		  //col_shape is not useful ..
		col2im_cpu(data_col,
				vol_shape[0], vol_shape[1], vol_shape[2], // channels, height, width
				kernel_shape[0], kernel_shape[1],	// kernel_h, kernel_w
				pad[0],    pad[1],		// pad_h, pad_w
				stride[0], stride[1],	// stride_h, stride_w
				data_vol, conv_out_factor );
	  } else {
	      const bool kVol2Col = false;
		  vol2col_core_cpu(data_col, kVol2Col, num_spatial_axes, vol_shape, col_shape,
				  kernel_shape, pad, stride, data_vol, conv_out_factor);
	  }
}

template <typename Dtype>
void col2vol_cpu(const Dtype* data_col, const int num_spatial_axes,
    const int* vol_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    Dtype* data_vol, const int sample_num) {
	//
	int col_inc = 1;
	for ( int i=0; i<num_spatial_axes; ++i )
		col_inc *= col_shape[i+1];
	int vol_size = 1;
	for ( int i=0; i<num_spatial_axes+1; ++i )
		vol_size *= vol_shape[i];

	for ( int b=0; b<sample_num; ++b ) {
		col2vol_cpu_internal<Dtype>( data_col+col_inc*b, num_spatial_axes,
			    vol_shape, col_shape, kernel_shape, pad, stride,
			    data_vol+vol_size*b, sample_num );
	}

}

// Explicit instantiation
template void col2vol_cpu<float>(const float* data_col,
    const int num_spatial_axes,
    const int* vol_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    float* data_vol, const int sample_num);
template void col2vol_cpu<double>(const double* data_col,
    const int num_spatial_axes,
    const int* vol_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    double* data_vol, const int sample_num);


// ------------------------------------------------------------------------------------------------------


#ifndef CPU_ONLY


template<typename Dtype>
struct vol2col_gpu_kernel_gen {
	template<int m> using func_tpl = vol2col_gpu_kernel_wrapper<Dtype,m>;
};

template<typename Dtype>
void vol2col_gpu_caller_func ( int num_axes, vol2col_gpu_caller_ARGLIST_T ) {
	try {
		zeta::rec_func_tpl_caller_func<1,vol2col_gpu_caller_MAX_AXES, vol2col_gpu_kernel_gen<Dtype>::template func_tpl >(
                    num_axes, vol2col_gpu_caller_ARGLIST );
	} catch ( const zeta::rec_func_tpl_call_overflow& ) {
		LOG(FATAL) << "vol2col_gpu does not support num_axes = " << num_axes;
	}
}

template <typename Dtype>
void vol2col_gpu(const Dtype* data_vol, const int num_spatial_axes,
	    const int num_kernels, const int* vol_shape, const int* col_shape,
	    const int* kernel_shape, const int* pad, const int* stride,
	    Dtype* data_col, const int sample_num) {

  int col_inc = 1;
  {
	  vector<int> conv_out_vec( num_spatial_axes );
	  caffe_copy(num_spatial_axes,col_shape+1,conv_out_vec.data());
	  for ( int i=0; i<conv_out_vec.size(); ++i )
		  col_inc *= conv_out_vec[i];
  }
  int vol_size = 1;
  {
	  vector<int> vol_shape_vec( num_spatial_axes+1 );
	  caffe_copy(num_spatial_axes+1,vol_shape,vol_shape_vec.data());
	  for ( int i=0; i<vol_shape_vec.size(); ++i )
		  vol_size *= vol_shape_vec[i];
  }

  for ( int b=0; b<sample_num; ++b ) {
	  vol2col_gpu_caller_func<Dtype>( num_spatial_axes, num_kernels,
			  data_vol+vol_size*b, vol_shape, col_shape,
			  kernel_shape, pad, stride, data_col+col_inc*b, sample_num );
  }

}

// Explicit instantiation
template void vol2col_gpu<float>(const float* data_vol,
    const int num_spatial_axes, const int col_size,
    const int* vol_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    float* data_col, const int sample_num);
template void vol2col_gpu<double>(const double* data_vol,
    const int num_spatial_axes, const int col_size,
    const int* vol_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    double* data_col, const int sample_num);



template<typename Dtype>
struct col2vol_gpu_kernel_gen {
	template<int m> using func_tpl = col2vol_gpu_kernel_wrapper<Dtype,m>;
};

template<typename Dtype>
void col2vol_gpu_caller_func ( int num_axes, col2vol_gpu_caller_ARGLIST_T ) {
	try {
		zeta::rec_func_tpl_caller_func<1,col2vol_gpu_caller_MAX_AXES, col2vol_gpu_kernel_gen<Dtype>::template func_tpl >( 
                    num_axes, col2vol_gpu_caller_ARGLIST );
	} catch ( const zeta::rec_func_tpl_call_overflow& ) {
		LOG(FATAL) << "col2vol_gpu does not support num_axes = " << num_axes;
	}
}

template <typename Dtype>
void col2vol_gpu(const Dtype* data_col, const int num_spatial_axes,
    const int vol_size, const int* vol_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    Dtype* data_vol, const int sample_num ) {

  int col_inc = 1;
  {
	  vector<int> conv_out_vec( num_spatial_axes );
	  caffe_copy(num_spatial_axes,col_shape+1,conv_out_vec.data());
	  for ( int i=0; i<conv_out_vec.size(); ++i )
		  col_inc *= conv_out_vec[i];
  }

  for ( int b=0; b<sample_num; ++b ) {
	  col2vol_gpu_caller_func<Dtype>( num_spatial_axes, vol_size, data_col+col_inc*b, vol_shape, col_shape,
			  kernel_shape, pad, stride, data_vol+vol_size*b, sample_num );
  }
}

// Explicit instantiation
template void col2vol_gpu<float>(const float* data_col,
    const int num_spatial_axes, const int vol_size,
    const int* vol_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    float* data_vol, const int sample_num);
template void col2vol_gpu<double>(const double* data_col,
    const int num_spatial_axes, const int vol_size,
    const int* vol_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    double* data_vol, const int sample_num);



#endif  // !CPU_ONLY



template <typename Dtype>
void col2im_nd_cpu(const Dtype* data_col, const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    Dtype* data_im) {
  const bool kIm2Col = false;
  im2col_nd_core_cpu(data_col, kIm2Col, num_spatial_axes, im_shape, col_shape,
                     kernel_shape, pad, stride, data_im);
}

// Explicit instantiation
template void col2im_nd_cpu<float>(const float* data_col,
    const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    float* data_im);
template void col2im_nd_cpu<double>(const double* data_col,
    const int num_spatial_axes,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    double* data_im);


}  // namespace caffe
