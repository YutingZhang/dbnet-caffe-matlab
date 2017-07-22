#include <vector>

#include "caffe/layers/tile_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void Tile(const int nthreads, const Dtype* bottom_data,
    const int tile_size, const int num_tiles, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int d = index % tile_size;
    const int n = index / tile_size / num_tiles;
    const int bottom_index = n * tile_size + d;
    top_data[index] = bottom_data[bottom_index];
  }
}

template <typename Dtype>
__global__ void TileC(const int nthreads, const Dtype* bottom_data,
    const int tile_size, const int num_tiles, const Dtype* scal_coeff, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
	const int d = index % tile_size;
	const int m = index / tile_size;
	const int n = m / num_tiles;
	const int t = m % num_tiles;
	const int bottom_index = n * tile_size + d;
	top_data[index] = scal_coeff[t] * bottom_data[bottom_index];
  }
}

template <typename Dtype>
void TileLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int nthreads = top[0]->count();
  if (has_coeff_)
	  TileC<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
		  <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
		  nthreads, bottom_data, inner_dim_, tiles_, coeff_.gpu_data(), top_data);
  else
	  Tile<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
		  <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
		  nthreads, bottom_data, inner_dim_, tiles_, top_data);
}

template <typename Dtype>
__global__ void TileBackward(const int nthreads, const Dtype* top_diff,
    const int tile_size, const int num_tiles, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int d = index % tile_size;
    const int n = (index / tile_size);
    bottom_diff[index] = 0;
    int top_index = n * num_tiles * tile_size + d;
    for (int t = 0; t < num_tiles; ++t) {
      bottom_diff[index] += top_diff[top_index];
      top_index += tile_size;
    }
  }
}

template <typename Dtype>
__global__ void TileCBackward(const int nthreads, const Dtype* top_diff,
    const int tile_size, const int num_tiles, const Dtype* scal_coeff, Dtype* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
	const int d = index % tile_size;
	const int n = (index / tile_size);
	bottom_diff[index] = 0;
	int top_index = n * num_tiles * tile_size + d;
	for (int t = 0; t < num_tiles; ++t) {
	  bottom_diff[index] += scal_coeff[t] * top_diff[top_index];
	  top_index += tile_size;
	}
  }
}

template <typename Dtype>
void TileLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int nthreads = bottom[0]->count();
  if (has_coeff_)
	  TileCBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
		  <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
		  nthreads, top_diff, inner_dim_, tiles_, coeff_.gpu_data(), bottom_diff);
  else
	  TileBackward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
		  <<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
		  nthreads, top_diff, inner_dim_, tiles_, bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(TileLayer);

}  // namespace caffe
