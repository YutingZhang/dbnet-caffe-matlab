#ifndef CAFFE_IMAGEDB_DATA_LAYER_HPP_
#define CAFFE_IMAGEDB_DATA_LAYER_HPP_

#include "boost/scoped_ptr.hpp"
#include "boost/shared_ptr.hpp"
#include "boost/thread.hpp"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/util/fake_leveldb.hpp"

#include <fstream>
#include <vector>

namespace caffe {

// This function is used to create a pthread that prefetches the data.
// Custom data layer, should keep it updated with the original one
template <typename Dtype>
void* ImagedbDataLayerPrefetch(void* layer_pointer);

struct ImagedbDataLayer_Inner;

template <typename Dtype>
class ImagedbDataLayer : public Layer<Dtype> {
    // The function used to perform prefetching.
    friend void* ImagedbDataLayerPrefetch<Dtype>(void* layer_pointer);

    shared_ptr<ImagedbDataLayer_Inner> a_;

    shared_ptr<int> instance_holder_;
    int instance_id_;

public:
    explicit ImagedbDataLayer(const LayerParameter& param);
    virtual ~ImagedbDataLayer();
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top) {};

    virtual inline const char* type() const { return "ImagedbData"; }
    virtual inline int ExactNumBottomBlobs() const { return 0; }
    virtual inline int MinTopBlobs() const { return 1; }
    virtual inline int MaxTopBlobs() const { return 3; }

protected:
    template< class DeviceMode >
    void Forward_generic(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);

    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    shared_ptr<caffe::fake_leveldb> db_;
    shared_ptr<caffe::fake_leveldb::Iterator> iter_;
    int datum_channels_;
    int color_type_;
    int max_buffered_files_;
    bool shuffle_;
    int global_shuffle_buffer_; // # batch
    vector< shared_ptr< vector<Dtype> > > gs_buffer_;
    pthread_t thread_;
    shared_ptr<Blob<Dtype> > prefetch_data_;
    shared_ptr<Blob<Dtype> > prefetch_mean_;
    shared_ptr<Blob<Dtype> > prefetch_label_;
    Blob<Dtype> data_mean_;

    virtual void pre_forward();
    virtual shared_ptr<caffe::fake_leveldb::Iterator> get_dataset_iter();
};

}

#endif

