// Copyright 2013 Yangqing Jia

#ifdef USE_OPENCV

#include <stdint.h>
#include <pthread.h>

#include <string>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/layers/imagedb_data_layer.hpp"

#include <boost/thread/mutex.hpp>
#include <map>
#include <algorithm>

#include <random>

using std::string;

namespace caffe {

struct ImagedbDataLayer_Inner {
    std::random_device rand_d;
    std::mt19937 rand_g;

    ImagedbDataLayer_Inner() : rand_g( rand_d() ) {}
};

class ImagedbDataLayer_Aux {
private:
	static boost::mutex instance_holder_mutex_;
	static std::map< string, shared_ptr<int> > instance_holder_map_;
public:
	static shared_ptr<int> instance_holder( const string& tag ) {
		boost::mutex::scoped_lock lock(instance_holder_mutex_);
		shared_ptr<int>& instance_holder = instance_holder_map_[tag];
		if (!instance_holder)
			instance_holder.reset( new int );
		return instance_holder;
	}
};
boost::mutex ImagedbDataLayer_Aux::instance_holder_mutex_;
std::map< string, shared_ptr<int> > ImagedbDataLayer_Aux::instance_holder_map_;

template<typename Dtype>
ImagedbDataLayer<Dtype>::ImagedbDataLayer(const LayerParameter& param)
    : Layer<Dtype>(param), a_( new ImagedbDataLayer_Inner ) {
	const string& data_src_path = param.imagedb_data_param().source();
	instance_holder_ = ImagedbDataLayer_Aux::instance_holder(data_src_path);
	instance_id_ = int(instance_holder_.use_count())-2;
}

template <typename Dtype>
void* ImagedbDataLayerPrefetch(void* layer_pointer) {
  CHECK(layer_pointer);
  ImagedbDataLayer<Dtype>* layer = reinterpret_cast<ImagedbDataLayer<Dtype>*>(layer_pointer);
  CHECK(layer);

  CHECK(layer->prefetch_data_);
  Dtype* top_data = layer->prefetch_data_->mutable_cpu_data();
  Dtype* top_label = layer->prefetch_label_->mutable_cpu_data();
  Dtype* top_mean  = layer->prefetch_mean_->mutable_cpu_data();
  const Dtype scale = layer->layer_param_.imagedb_data_param().scale();
  const int batch_size = layer->layer_param_.imagedb_data_param().batch_size();
  const int crop_size = layer->layer_param_.imagedb_data_param().crop_size();
  const bool mirror = layer->layer_param_.imagedb_data_param().mirror();
  const Dtype color_jitter_std = layer->layer_param_.imagedb_data_param().color_jitter_std();
  const string color_jitter_profile = layer->layer_param_.imagedb_data_param().color_jitter_profile();

  if (mirror && crop_size == 0) {
    LOG(FATAL) << "Current implementation requires mirror and crop_size to be "
        << "set at the same time.";
  }

  // channels
  const int channels = layer->datum_channels_;
  bool cropped_mean = (layer->data_mean_.height()==crop_size &&
		  layer->data_mean_.width()==crop_size);

  // mean data
  const Dtype* mean = layer->data_mean_.cpu_data();

  for (int itemid = 0; itemid < batch_size; ++itemid) {

	// get an image
	std::pair<cv::Mat,int> ldata = layer->iter_->get_labeled_image();
	Datum datum;
	CVMatToDatum( ldata.first, &datum );
	datum.set_label(ldata.second);

	// datum size check
	const int height = datum.height();
	const int width = datum.width();
    CHECK_GE(height, crop_size) << "crop_size should not be larger than height. " <<
    		"crop_size: " << crop_size << "  height: " << height;
	CHECK_GE(width,  crop_size) << "crop_size should not be larger than width. " <<
    		"crop_size: " << crop_size << "  width:  " << width;
	CHECK( cropped_mean || (layer->data_mean_.height()>= height) );
	CHECK( cropped_mean || (layer->data_mean_.width() >= width ) );

	const int size = datum.channels() * datum.height() * datum.width();
	// mean size
	const int mean_height = layer->data_mean_.height();
	const int mean_width  = layer->data_mean_.width();
	const int mean_h_off  = (cropped_mean)? ( 0 ) : ( (mean_height-height)/2 );
	const int mean_w_off  = (cropped_mean)? ( 0 ) : ( (mean_width- width )/2 );

	/// color jittering
	vector<Dtype> bgr_color_shift({Dtype(0),Dtype(0),Dtype(0)});
	if ( color_jitter_std ) {
        CHECK(crop_size) << "color_jitter_std only works with crop_size";
		vector<Dtype> color_jitter_lambda(3,Dtype(0));
		vector<Dtype> color_jitter_bgr_vec(9,Dtype(0));
		if ( color_jitter_profile == "imagenet:pca" ) {
			color_jitter_lambda  = vector<Dtype>( {Dtype(13.3371),Dtype(0.4455),Dtype(0.0857)} );
			color_jitter_bgr_vec = vector<Dtype>(
					{   Dtype(-0.7808),Dtype(-0.5185),Dtype( 0.3485),
						Dtype(-0.6142),Dtype( 0.5348),Dtype(-0.5803),
						Dtype(-0.1145),Dtype( 0.6672),Dtype( 0.7360) } );
		} else {
			LOG(FATAL) << "unrecongized color jittering profiles";
		}
		bgr_color_shift = vector<Dtype>({ Dtype(randn_threadsafe()*color_jitter_std),
							Dtype(randn_threadsafe()*color_jitter_std),
							Dtype(randn_threadsafe()*color_jitter_std)});
		caffe_mul( 3, color_jitter_lambda.data(), bgr_color_shift.data(), color_jitter_lambda.data() );
		caffe_cpu_gemv(CblasNoTrans, 3, 3, Dtype(1.), color_jitter_bgr_vec.data(),
				color_jitter_lambda.data(), Dtype(0.), bgr_color_shift.data() );
	}

	///
    const string& data = datum.data();
    if (crop_size) {
      CHECK(data.size()) << "Image cropping only support uint8 data";
      int h_off = 0, w_off = 0;
      // We only do random crop when we do training.
      if (layer->phase_ == TRAIN) {
        // NOLINT_NEXT_LINE(runtime/threadsafe_fn)
    	if (height>crop_size)
    		h_off = rand() % (height - crop_size);
        // NOLINT_NEXT_LINE(runtime/threadsafe_fn)
    	if (width>crop_size)
    		w_off = rand() % (width - crop_size);
      } else {
        h_off = (height - crop_size) / 2;
        w_off = (width - crop_size) / 2;
      }
      // NOLINT_NEXT_LINE(runtime/threadsafe_fn)
      if (mirror && rand() % 2) {
        // Copy mirrored version
        for (int c = 0; c < channels; ++c) {
          for (int h = 0; h < crop_size; ++h) {
            for (int w = 0; w < crop_size; ++w) {
              Dtype mval = mean[(c * mean_height + h + (cropped_mean?0:(mean_h_off+h_off)) ) * mean_width
            		  + w + (cropped_mean?0:(mean_w_off+w_off)) ] + bgr_color_shift[c];
              int cur_idx = ((itemid * channels + c) * crop_size + h) * crop_size + crop_size - 1 - w;
              top_data[cur_idx] = (static_cast<Dtype>( (uint8_t)data[(c * height + h + h_off) * width
                          + w + w_off]) - mval) * scale;
              top_mean[cur_idx] = mval * scale;
            }
          }
        }
      } else {
        // Normal copy
        for (int c = 0; c < channels; ++c) {
          for (int h = 0; h < crop_size; ++h) {
            for (int w = 0; w < crop_size; ++w) {
              Dtype mval = mean[(c * mean_height + h + (cropped_mean?0:(mean_h_off+h_off)) ) * mean_width
            		  + w + (cropped_mean?0:(mean_w_off+w_off)) ] + bgr_color_shift[c];
              int cur_idx = ((itemid * channels + c) * crop_size + h) * crop_size + w;
              top_data[cur_idx] = (static_cast<Dtype>(
                      (uint8_t)data[(c * height + h + h_off) * width + w + w_off]) - mval) * scale;
              top_mean[cur_idx] = mval * scale;
            }
          }
        }
      }
    } else {
      // we will prefer to use data() first, and then try float_data()
      if (data.size()) {
        for (int j = 0; j < size; ++j) {
          top_data[itemid * size + j] =
              (static_cast<Dtype>((uint8_t)data[j]) - mean[j]) * scale;
          top_mean[itemid * size + j] = mean[j] * scale;
        }
      } else {
        for (int j = 0; j < size; ++j) {
          top_data[itemid * size + j] =
              (datum.float_data(j) - mean[j]) * scale;
          top_mean[itemid * size + j] = mean[j] * scale;
        }
      }
    }

    top_label[itemid] = datum.label();
    // go to the next iter
    layer->iter_->Next();
    if (!layer->iter_->Valid()) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      layer->iter_->SeekToFirst( layer->shuffle_ );
    }
  }

  return reinterpret_cast<void*>(NULL);
}

template <typename Dtype>
ImagedbDataLayer<Dtype>::~ImagedbDataLayer<Dtype>() {
  // Finally, join the thread
  CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
}

template <typename Dtype>
shared_ptr<caffe::fake_leveldb::Iterator> ImagedbDataLayer<Dtype>::get_dataset_iter() {
	caffe::size2d img_siz;
	auto& idparam = this->layer_param_.imagedb_data_param();
	if ( idparam.has_new_shorter() ) {
		CHECK( !idparam.has_new_width() && !idparam.has_new_height() ) <<
				"new_shorter cannot be specified together with new_width or new_height";
		img_siz = caffe::size2d(
				this->layer_param_.imagedb_data_param().new_shorter(),
				caffe::size2d::shorter() );
		if ( idparam.has_new_shorter_max() )
			img_siz.s_max = this->layer_param_.imagedb_data_param().new_shorter();
	} else {
		img_siz = caffe::size2d( idparam.new_width(), idparam.new_height());
	}
	if ( idparam.has_new_rot_max() ) {
		img_siz.rot_max = this->layer_param_.imagedb_data_param().new_rot_max();
	}
	img_siz.s_min = this->layer_param_.imagedb_data_param().crop_size();
	img_siz.s_level = this->layer_param_.imagedb_data_param().new_shorter_level();
	img_siz.rot_level = this->layer_param_.imagedb_data_param().new_rot_level();
	shared_ptr<caffe::fake_leveldb::Iterator> iter(
			db_->NewIterator(max_buffered_files_, img_siz, color_type_));
	iter->SeekToFirst();
	return iter;
}

template <typename Dtype>
void ImagedbDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 0) << "Data Layer takes no input blobs.";
  CHECK_LE(top.size(), 3) << "Data Layer takes at most three blobs as output.";
  CHECK_GE(top.size(), 1) << "Data Layer takes at least one blob as output.";

  //
  shuffle_ = this->layer_param_.imagedb_data_param().shuffle();
  global_shuffle_buffer_ = this->layer_param_.imagedb_data_param().global_shuffle_buffer();

  // Initialize the leveldb
  caffe::fake_leveldb* db_temp = NULL;
  std::string source_type = this->layer_param_.imagedb_data_param().source_type();
  max_buffered_files_ = this->layer_param_.imagedb_data_param().max_open_files();
  if ( source_type == "imagedb" ) {
	  if ( max_buffered_files_<=0 ) max_buffered_files_ = 2;
	  db_temp = new caffe::fake_leveldb_imagedb;
  } else if ( source_type == "bbox" ) {
	  if ( max_buffered_files_<=0 ) max_buffered_files_ = 20;
	  db_temp = new caffe::fake_leveldb_bbox;
  } else {
	  LOG(ERROR) << "Opening fake leveldb ( Type: "
			  << source_type << ") " << this->layer_param_.imagedb_data_param().source();
  }

  std::string color_type_str = this->layer_param_.imagedb_data_param().color_type();
  color_type_ = caffe::fake_leveldb::BGR_COLOR;
  if ( color_type_str == "color" ) {
	  color_type_ = caffe::fake_leveldb::BGR_COLOR ;
  } else if ( color_type_str == "grayscale" ) {
	  color_type_ = caffe::fake_leveldb::GRAY_SCALE ;
  } else if ( color_type_str == "as_is" ) {
	  color_type_ = caffe::fake_leveldb::AS_IS_COLOR ;
  } else {
	  LOG(ERROR) << "Unknown color type: " << color_type_str ;
  }

  LOG(INFO) << "Opening fake leveldb ( Type: "
			  << source_type << ") " << this->layer_param_.imagedb_data_param().source();
  int status = db_temp->OpenReadOnly( this->layer_param_.imagedb_data_param().source() ) ;

  CHECK(~status) << "Failed to open fake leveldb "
      << this->layer_param_.imagedb_data_param().source() << std::endl
      << "  Error code : " << status << std::endl;
  /*
  leveldb::DB* db_temp;
  leveldb::Options options;
  options.create_if_missing = false;
  options.max_open_files = 100;
  LOG(INFO) << "Opening leveldb " << this->layer_param_.imagedb_data_param().source();
  leveldb::Status status = leveldb::DB::Open(
      options, this->layer_param_.imagedb_data_param().source(), &db_temp);
  CHECK(status.ok()) << "Failed to open leveldb "
      << this->layer_param_.imagedb_data_param().source() << std::endl << status.ToString();
  */
  db_.reset(db_temp);
  auto iter = get_dataset_iter();
  iter_.reset();
  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  std::pair<cv::Mat,int> ldata = iter->get_labeled_image();
  CVMatToDatum( ldata.first, &datum );
  datum.set_label(ldata.second);
  // image
  int crop_size = this->layer_param_.imagedb_data_param().crop_size();
  if (crop_size > 0) {
    top[0]->Reshape(
        this->layer_param_.imagedb_data_param().batch_size(), datum.channels(), crop_size, crop_size);
    if (top.size()>=3)
        top[2]->Reshape(
            this->layer_param_.imagedb_data_param().batch_size(), datum.channels(), crop_size, crop_size);
    prefetch_data_.reset(new Blob<Dtype>(
        this->layer_param_.imagedb_data_param().batch_size(), datum.channels(), crop_size, crop_size));
    prefetch_mean_.reset(new Blob<Dtype>(
        this->layer_param_.imagedb_data_param().batch_size(), datum.channels(), crop_size, crop_size));
  } else {
    top[0]->Reshape(
        this->layer_param_.imagedb_data_param().batch_size(), datum.channels(), datum.height(),
        datum.width());
    if (top.size()>=3)
        top[2]->Reshape(
            this->layer_param_.imagedb_data_param().batch_size(), datum.channels(), datum.height(),
            datum.width());
    prefetch_data_.reset(new Blob<Dtype>(
        this->layer_param_.imagedb_data_param().batch_size(), datum.channels(), datum.height(),
        datum.width()));
    prefetch_mean_.reset(new Blob<Dtype>(
        this->layer_param_.imagedb_data_param().batch_size(), datum.channels(), datum.height(),
        datum.width()));
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (top.size()>1)
	  top[1]->Reshape(this->layer_param_.imagedb_data_param().batch_size(), 1, 1, 1);
  prefetch_label_.reset(
      new Blob<Dtype>(this->layer_param_.imagedb_data_param().batch_size(), 1, 1, 1));
  // datum size
  datum_channels_ = datum.channels();
  // check if we want to have mean
  if (this->layer_param_.imagedb_data_param().has_mean_file()) {
    BlobProto blob_proto;
    LOG(INFO) << "Loading mean file from " << this->layer_param_.imagedb_data_param().mean_file();
    ReadProtoFromBinaryFile(this->layer_param_.imagedb_data_param().mean_file().c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
    CHECK_EQ(data_mean_.num(), 1);
    CHECK_EQ(data_mean_.channels(), datum_channels_);
  } else {
    // Simply initialize a const mean.
	CHECK_GT(crop_size,0) << "crop_size must be greater than 0, when using const mean";
    data_mean_.Reshape(1, datum_channels_, crop_size, crop_size);

    Dtype mean_value[datum_channels_];

	int m_channels = this->layer_param_.imagedb_data_param().mean_values_size();
	for (int i=0; i<datum_channels_; ++i) {
		if (m_channels>i)
			mean_value[i] = this->layer_param_.imagedb_data_param().mean_values(i);
		else
			mean_value[i] = this->layer_param_.imagedb_data_param().mean_value();
	}

    //Dtype mean_value = this->layer_param_.imagedb_data_param().mean_value();
    Dtype* p0 = data_mean_.mutable_cpu_data();
    const int height = crop_size, width = crop_size;

    for (int i=0; i<datum_channels_; ++i) {
    	Dtype * p = p0 + data_mean_.offset(0,i,0,0);
		Dtype * const q = p + height*width;
		while (p<q)
			*(p++) = mean_value[i];
    }
  }
  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  prefetch_data_->mutable_cpu_data();
  prefetch_mean_->mutable_cpu_data();
  prefetch_label_->mutable_cpu_data();
  data_mean_.mutable_cpu_data();

  if ( global_shuffle_buffer_>0 ) {
	  gs_buffer_.resize( top.size() );
	  for ( size_t i = 0; i<top.size(); ++i ) {
		  size_t s = top[i]->count();
		  s *= global_shuffle_buffer_;
		  gs_buffer_[i].reset( new vector<Dtype>( s ) );
	  }
  }
}

template <typename Dtype>
template< class DeviceMode >
void ImagedbDataLayer<Dtype>::Forward_generic(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  pre_forward();
  // First, join the thread
  CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
  // Copy the data
  if ( global_shuffle_buffer_>0 ) {
	  int batch_size = top[0]->shape(0);
	  int buffer_size = batch_size * global_shuffle_buffer_;
	  vector<int> t(buffer_size);
	  for ( int i=0; i<buffer_size; ++i )
		  t[i] = i;
	  std::shuffle( t.begin(), t.end(), this->a_->rand_g );

	  Dtype* top_data = NULL,* top_label = NULL,* top_mean = NULL;
	  Dtype* gsb_data = NULL,* gsb_label = NULL,* gsb_mean = NULL;
	  int data_siz = 0, label_siz = 0, mean_siz = 0;
	  top_data = top[0]->mutable_data( DeviceMode() );
	  gsb_data = gs_buffer_[0]->data();
	  data_siz = top[0]->count(1);
	  if (top.size()>1) {
		  top_label = top[1]->mutable_data( DeviceMode() );
		  gsb_label = gs_buffer_[1]->data();
		  label_siz = top[1]->count(1);
	  }
	  if (top.size()>2) {
		  top_mean = top[2]->mutable_data( DeviceMode() );
		  gsb_mean = gs_buffer_[2]->data();
		  mean_siz = top[2]->count(1);
	  }
	  for ( int i=0; i<batch_size; ++i ) {
		  int j = t[i];
		  caffe_copy( data_siz, gsb_data + j*data_siz, top_data + i*data_siz );
		  caffe_copy( data_siz, prefetch_data_->cpu_data() + i*data_siz, gsb_data + j*data_siz );
		  if (top.size()>1) {
			  caffe_copy( label_siz, gsb_label + j*label_siz, top_label + i*label_siz );
			  caffe_copy( label_siz, prefetch_label_->cpu_data() + i*label_siz, gsb_label + j*label_siz );
		  }
		  if (top.size()>2) {
			  caffe_copy( mean_siz, gsb_mean + j*mean_siz, top_mean + i*mean_siz );
			  caffe_copy( mean_siz, prefetch_mean_->cpu_data() + i*mean_siz, gsb_mean + j*mean_siz );
		  }
	  }
  } else {
	  caffe_copy( prefetch_data_->count(), prefetch_data_->cpu_data(), top[0]->mutable_data( DeviceMode() ) );
	  if (top.size()>1)
		  caffe_copy( prefetch_label_->count(), prefetch_label_->cpu_data(), top[1]->mutable_data( DeviceMode() ) );
	  if (top.size()>2)
		  caffe_copy(prefetch_mean_->count(), prefetch_mean_->cpu_data(), top[2]->mutable_data( DeviceMode() ) );
  }

  // Start a new prefetch thread
  CHECK(!pthread_create(&thread_, NULL, ImagedbDataLayerPrefetch<Dtype>,
      reinterpret_cast<void*>(this))) << "Pthread execution failed.";
}

template <typename Dtype>
void ImagedbDataLayer<Dtype>::pre_forward() {

  if ( iter_ ) return;
  {
	// Open dataset
	LOG(INFO) << "Reopen fake_leveldb";
	iter_ = get_dataset_iter();
	if (shuffle_) iter_->SeekToFirst( shuffle_ );
	// Check if we would need to randomly skip a few data points
	unsigned int skip = 0;
	unsigned int dataset_size = iter_->Total();
	if (this->layer_param_.imagedb_data_param().rand_skip()) {
		// NOLINT_NEXT_LINE(runtime/threadsafe_fn)
		unsigned int rand_skip_range = this->layer_param_.imagedb_data_param().rand_skip();
		rand_skip_range = std::min(rand_skip_range, dataset_size);
		skip = rand() % rand_skip_range;
	}
	int instance_total = int(instance_holder_.use_count())-1;
	LOG(INFO) << "Instance ID : " << instance_id_ << " / " << instance_total;
	skip += (int)std::floor( double(dataset_size)/double(instance_total)*instance_id_ );
	skip %= dataset_size;
	LOG(INFO) << "Skipping first " << skip << " data points.";
	while (skip-- > 0) {
		iter_->Next();
		if (!iter_->Valid()) {
		  iter_->SeekToFirst();	// no need to re-shuffle here
		}
	}
	DLOG(INFO) << "Initializing prefetch";
	CHECK(!pthread_create(&thread_, NULL, ImagedbDataLayerPrefetch<Dtype>,
	  reinterpret_cast<void*>(this))) << "Pthread execution failed.";
	DLOG(INFO) << "Prefetch initialized.";

	if ( global_shuffle_buffer_>0 ) {
		LOG(INFO) << "Prefetch for global random shuffling";
		for ( int i=0; i<global_shuffle_buffer_; ++i ) {
			// First, join the thread
			CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
			// Copy the data
			caffe_copy( prefetch_data_->count(), prefetch_data_->cpu_data(),
					gs_buffer_[0]->data() + prefetch_data_->count() * i );
			if (gs_buffer_.size()>1)
				caffe_copy( prefetch_label_->count(), prefetch_label_->cpu_data(),
						gs_buffer_[1]->data() + prefetch_label_->count() * i );
			if (gs_buffer_.size()>2)
				caffe_copy(prefetch_mean_->count(), prefetch_mean_->cpu_data(),
						gs_buffer_[2]->data() + prefetch_mean_->count() * i );
			// Start a new prefetch thread
			CHECK(!pthread_create(&thread_, NULL, ImagedbDataLayerPrefetch<Dtype>,
							reinterpret_cast<void*>(this))) << "Pthread execution failed.";
		}
	}

  }

}

template<typename Dtype>
void ImagedbDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	Forward_generic<mCPU>(bottom,top);
}



#ifdef CPU_ONLY
STUB_GPU(ImagedbDataLayer);
#else

template<typename Dtype>
void ImagedbDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	Forward_generic<mGPU>(bottom,top);
}

template<typename Dtype>
void ImagedbDataLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) { }

#endif

INSTANTIATE_CLASS(ImagedbDataLayer);
REGISTER_LAYER_CLASS(ImagedbData);


}  // namespace caffe

#endif
