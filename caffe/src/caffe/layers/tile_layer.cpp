#include <vector>

#include "caffe/layers/tile_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using namespace std;

template <typename Dtype>
void TileLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const TileParameter& tile_param = this->layer_param_.tile_param();
  axis_ = bottom[0]->CanonicalAxisIndex(tile_param.axis());
  CHECK(tile_param.has_tiles()) << "Number of tiles must be specified";
  tiles_ = tile_param.tiles();
  CHECK_GT(tiles_, 0) << "Number of tiles must be positive.";
  vector<int> top_shape = bottom[0]->shape();
  top_shape[axis_] = bottom[0]->shape(axis_) * tiles_;
  top[0]->Reshape(top_shape);
  outer_dim_ = bottom[0]->count(0, axis_);
  inner_dim_ = bottom[0]->count(axis_);

  if ( tile_param.coeff_size() ) {
	  CHECK_EQ( tile_param.coeff_size(), tiles_ ) << "dimension of coeff should be the same as tiles";
	  coeff_.Reshape( { tile_param.coeff_size() } );
	  Dtype* coeff_data = coeff_.mutable_cpu_data();
	  for ( unsigned int i = 0; i<tiles_; ++i )
		  coeff_data[i] = tile_param.coeff(i);
	  has_coeff_ = true;
  } else {
	  has_coeff_ = false;
  }
}

template <typename Dtype>
void TileLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* coeff_data = (has_coeff_) ? coeff_.cpu_data() : NULL;
  for (int i = 0; i < outer_dim_; ++i) {
	for (int t = 0; t < tiles_; ++t) {
	  if ( coeff_data )
		  caffe_cpu_axpby(inner_dim_, coeff_data[t], bottom_data, Dtype(0), top_data);
	  else
		  caffe_copy(inner_dim_, bottom_data, top_data);
	  top_data += inner_dim_;
	}
	bottom_data += inner_dim_;
  }
}

template <typename Dtype>
void TileLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* coeff_data = (has_coeff_) ? coeff_.cpu_data() : NULL;
  for (int i = 0; i < outer_dim_; ++i) {
    caffe_copy(inner_dim_, top_diff, bottom_diff);
    top_diff += inner_dim_;
    for (int t = 1; t < tiles_; ++t) {
      Dtype tscl = ( coeff_data ) ? coeff_data[t] : Dtype(1);
      caffe_axpy(inner_dim_, tscl, top_diff, bottom_diff);
      top_diff += inner_dim_;
    }
    bottom_diff += inner_dim_;
  }
}

#ifdef CPU_ONLY
STUB_GPU(TileLayer);
#endif

INSTANTIATE_CLASS(TileLayer);
REGISTER_LAYER_CLASS(Tile);

}  // namespace caffe
