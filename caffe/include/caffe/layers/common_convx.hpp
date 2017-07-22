#ifndef CAFFE_COMMON_CONVX_HPP_
#define CAFFE_COMMON_CONVX_HPP_

#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "boost/shared_ptr.hpp"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template<typename Dtype>
struct CommonInternalBottomTop {

    int channel_axis_;

        vector<Blob<Dtype>*> bottom_cag_;
        vector<Blob<Dtype>*> top_cag_;
        vector<shared_ptr< Blob<Dtype> > > bottom_cag0_;
        vector<shared_ptr< Blob<Dtype> > > top_cag0_;

        enum TopUpdateLevel {
                TOP_NONE = 0,
                TOP_SHAPE,
                TOP_FULL
        };

        void bottom_top_channel_as_geo( const vector<Blob<Dtype>*>& bottom,
                        const vector<Blob<Dtype>*>& top, int update_internal_top = TOP_NONE );
        void data_channel_as_geo( const vector<Blob<Dtype>*>& input,
                        vector<Blob<Dtype>*>& output, vector<shared_ptr< Blob<Dtype> > >& output0,
                        int channel_offset );

        void bottom_top_geo_as_channel( const vector<Blob<Dtype>*>& bottom,
                        const vector<Blob<Dtype>*>& top, int update_internal_top = TOP_NONE );
        void data_geo_as_channel( const vector<Blob<Dtype>*>& input,
                        vector<Blob<Dtype>*>& output, vector<shared_ptr< Blob<Dtype> > >& output0,
                        int channel_offset );


};

// De/Convolution with Channel-wise Reweighting
enum ConvLayer_TYPE {
        ConvLayer_T,
        DeconvLayer_T,
        ndConvLayer_T,
        ndDeconvLayer_T
};

template <typename Dtype, int ConvOrDeconv_T>
class CommonConvolutionLayer {};

/*
template <typename Dtype>
class CommonConvolutionLayer<Dtype,ConvLayer_T> : public ConvolutionLayer<Dtype> {
public:
  typedef ConvolutionLayer<Dtype> layer_t;
  explicit CommonConvolutionLayer(const LayerParameter& param)
          : ConvolutionLayer<Dtype>(param) {}
};

template <typename Dtype>
class CommonConvolutionLayer<Dtype,DeconvLayer_T> : public DeconvolutionLayer<Dtype> {
public:
  typedef DeconvolutionLayer<Dtype> layer_t;
  explicit CommonConvolutionLayer(const LayerParameter& param)
          : DeconvolutionLayer<Dtype>(param) {}
};
*/


template <typename Dtype, int ConvOrDeconv_T>
class CommonConvolutionLayerX : public CommonConvolutionLayer<Dtype,ConvOrDeconv_T> {
public:
    typedef CommonConvolutionLayer<Dtype,ConvOrDeconv_T> common_conv_t;
    typedef CommonConvolutionLayerX<Dtype,ConvOrDeconv_T> this_t;
protected:
    bool fake_channel_; // for taking channel as conv
    CommonInternalBottomTop<Dtype> conv_ibt;
    shared_ptr<this_t> internal_conv_layer_;
public:
    explicit CommonConvolutionLayerX(const LayerParameter& param, bool fake_channel = false) 
        : common_conv_t(param), fake_channel_(fake_channel) {
    }
        void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
          const vector<Blob<Dtype>*>& top, bool fake_channel );
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
          const vector<Blob<Dtype>*>& top ) { LayerSetUp(bottom,top,fake_channel_); }
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
          const vector<Blob<Dtype>*>& top);
protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
          const vector<Blob<Dtype>*>& top);
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
          const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
          const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
          const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
public:
        virtual void pub_Forward_cpu(const vector<Blob<Dtype>*>& bottom,
          const vector<Blob<Dtype>*>& top) {
                Forward_cpu( bottom, top );
        }
        virtual void pub_Forward_gpu(const vector<Blob<Dtype>*>& bottom,
          const vector<Blob<Dtype>*>& top) {
                Forward_gpu( bottom, top );
        }
        virtual void pub_Backward_cpu(const vector<Blob<Dtype>*>& top,
          const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
                Backward_cpu(top, propagate_down, bottom);
        }
        virtual void pub_Backward_gpu(const vector<Blob<Dtype>*>& top,
          const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
                Backward_gpu(top, propagate_down, bottom);
        }
};

}

#endif

