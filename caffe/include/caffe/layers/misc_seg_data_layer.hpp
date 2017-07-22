/*
 * misc_seg_data_layer.hpp
 *
 *  Created on: Jun 11, 2014
 *      Author: zhangyuting
 */

#ifndef CAFFE_MISC_SEG_DATA_HPP_
#define CAFFE_MISC_SEG_DATA_HPP_

#include "boost/scoped_ptr.hpp"
#include "boost/shared_ptr.hpp"
#include "boost/thread.hpp"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_data_layer.hpp"

#include <fstream>
#include <vector>

namespace caffe {

//////////////////////////

/**
 * A minimal wrapper for boost::thread to force host compilation for boost
 * Defined in caffe/util/thread.hpp
 */

class Thread_Fallback {
 public:
  template<typename Callable, class A1>
  Thread_Fallback(Callable func, A1 a1);
  void join();
  bool joinable();
 private:
  void* thread_;
};


/**
 * Virtual class encapsulate boost::thread for use in base class
 * The child class will acquire the ability to run a single thread,
 * by reimplementing the virutal function InternalThreadEntry.
 */
class InternalThread_Fallback {
 public:
  InternalThread_Fallback() : thread_(NULL) {}
  virtual ~InternalThread_Fallback();

  /** Returns true if the thread was successfully started. **/
  bool StartInternalThread();

  /** Will not return until the internal thread has exited. */
  bool WaitForInternalThreadToExit();

  bool is_started() const { return thread_ != NULL && thread_->joinable(); }

 protected:
  /* Implement this method in your subclass
      with the code you want your thread to run. */
  virtual void InternalThreadEntry() {}

  caffe::Thread_Fallback* thread_;
};

template <typename Dtype>
class BasePrefetchingDataLayer_Fallback :
    public BaseDataLayer<Dtype>, public InternalThread_Fallback {
 public:
  explicit BasePrefetchingDataLayer_Fallback(const LayerParameter& param)
      : BaseDataLayer<Dtype>(param) {}
  virtual ~BasePrefetchingDataLayer_Fallback() {}
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden.
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void CreatePrefetchThread();
  virtual void JoinPrefetchThread();
  // The thread's function
  virtual void InternalThreadEntry() {}

 protected:
  Blob<Dtype> prefetch_data_;
  Blob<Dtype> prefetch_label_;
  Blob<Dtype> transformed_data_;
};

/** Jay add
 * @brief prefetching data layer which also prefetches data dimensions
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class ImageDimPrefetchingDataLayer : public BasePrefetchingDataLayer_Fallback<Dtype> {
/*
 notice:
 this code is based on the following implementation.
 https://bitbucket.org/deeplab/deeplab-public/
 */
 public:
  explicit ImageDimPrefetchingDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer_Fallback<Dtype>(param) {}
  virtual ~ImageDimPrefetchingDataLayer() {}
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden.
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  // The thread's function
  virtual void InternalThreadEntry() {}

 protected:
  Blob<Dtype> prefetch_data_dim_;
  bool output_data_dim_;
};

template <typename Dtype>
class ImageSegDataLayer : public ImageDimPrefetchingDataLayer<Dtype> {
 public:
  explicit ImageSegDataLayer(const LayerParameter& param)
    : ImageDimPrefetchingDataLayer<Dtype>(param) {}
  virtual ~ImageSegDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const {
    return "ImageSegData";
  }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 3; }
  virtual inline bool AutoTopBlobs() const { return true; }

 protected:
  virtual void ShuffleImages();
  virtual void InternalThreadEntry();

 protected:
  Blob<Dtype> transformed_label_;

  shared_ptr<Caffe::RNG> prefetch_rng_;

  vector<std::pair<std::string, std::string> > lines_;
  int lines_id_;
};


template <typename Dtype>
class WindowSegDataLayer : public ImageDimPrefetchingDataLayer<Dtype> {
 public:
  explicit WindowSegDataLayer(const LayerParameter& param)
    : ImageDimPrefetchingDataLayer<Dtype>(param) {}
  virtual ~WindowSegDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const {
    return "WindowSegDataLayer";
  }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 3; }
  virtual inline bool AutoTopBlobs() const { return true; }

 protected:
  virtual void ShuffleImages();
  virtual void InternalThreadEntry();

 protected:
  Blob<Dtype> transformed_label_;

  shared_ptr<Caffe::RNG> prefetch_rng_;

  typedef struct SegItems {
    std::string imgfn;
    std::string segfn;
    int x1, y1, x2, y2;
  } SEGITEMS;

  vector<SEGITEMS> lines_;
  int lines_id_;
};


}

#endif

