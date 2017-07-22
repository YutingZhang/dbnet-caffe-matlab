#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/im2col_branched.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe_branched {

/*
template <typename Dtype, int num_axes>
__global__ void vol2col_gpu_kernel(const int n, const Dtype* data_im,
    const int* im_shape, const int* col_shape,
    const int* kernel_shape, const int* pad, const int* stride,
    Dtype* data_col, const int);
    */

template <typename Dtype, int num_axes>
__global__ void vol2col_gpu_kernel(vol2col_gpu_caller_ARGLIST_T);

}

namespace caffe {

using namespace caffe_branched;

// Forward declare kernel functions

extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;

template <typename Dtype>
class Vol2colKernelTest : public GPUDeviceTest<Dtype> {
 protected:
  Vol2colKernelTest()
        // big so launches > 1024 threads
      : blob_bottom_(new Blob<Dtype>(5, 500, 10, 10)),
        blob_kernel_shape_(new Blob<int>()),
        blob_stride_(new Blob<int>()),
        blob_pad_(new Blob<int>()),
        blob_top_(new Blob<Dtype>()),
        blob_top_cpu_(new Blob<Dtype>()) {
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    vector<int> dim_blob_shape(1, 2);
    blob_kernel_shape_->Reshape(dim_blob_shape);
    blob_stride_->Reshape(dim_blob_shape);
    blob_pad_->Reshape(dim_blob_shape);

    height_ = blob_bottom_->height();
    width_ = blob_bottom_->width();
    channels_ = blob_bottom_->channels();
    pad_ = 0;
    stride_ = 2;
    kernel_size_ = 3;
    height_col_ = (height_ + 2 * pad_ - kernel_size_) / stride_ + 1;
    width_col_ = (width_ + 2 * pad_ - kernel_size_) / stride_ + 1;

    for (int i = 0; i < 2; ++i) {
      blob_kernel_shape_->mutable_cpu_data()[i] = kernel_size_;
      blob_stride_->mutable_cpu_data()[i] = stride_;
      blob_pad_->mutable_cpu_data()[i] = pad_;
    }
  }

  virtual ~Vol2colKernelTest() {
    delete blob_bottom_;
    delete blob_top_;
    delete blob_top_cpu_;
    delete blob_kernel_shape_;
    delete blob_stride_;
    delete blob_pad_;
  }

  Blob<int>* const blob_kernel_shape_;
  Blob<int>* const blob_stride_;
  Blob<int>* const blob_pad_;
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_cpu_;
  int height_;
  int width_;
  int channels_;
  int pad_;
  int stride_;
  int kernel_size_;
  int height_col_;
  int width_col_;
};

TYPED_TEST_CASE(Vol2colKernelTest, TestDtypes);

TYPED_TEST(Vol2colKernelTest, TestGPU) {
  // Reshape the blobs to correct size for vol2col output
  this->blob_top_->Reshape(this->blob_bottom_->num(),
      this->channels_ * this->kernel_size_ * this->kernel_size_,
      this->height_col_,
      this->width_col_);

  this->blob_top_cpu_->ReshapeLike(*this->blob_top_);

  const TypeParam* bottom_data_cpu = this->blob_bottom_->cpu_data();
  TypeParam* top_data_cpu = this->blob_top_cpu_->mutable_cpu_data();

  // CPU Version
  for (int n = 0; n < this->blob_bottom_->num(); ++n) {
    vol2col_cpu(bottom_data_cpu + this->blob_bottom_->offset(n), 2,
        this->blob_bottom_->shape().data() + 1,
        this->blob_top_cpu_->shape().data() + 1,
        this->blob_kernel_shape_->cpu_data(),
        this->blob_pad_->cpu_data(), this->blob_stride_->cpu_data(),
        top_data_cpu + this->blob_top_cpu_->offset(n));
  }

  // GPU version
  int num_kernels = this->channels_ * this->height_col_ * this->width_col_;
  int default_grid_dim = CAFFE_GET_BLOCKS(num_kernels);
  const TypeParam* bottom_data_gpu = this->blob_bottom_->gpu_data();

  // Launch with different grid sizes
  for (int grid_div = 2; grid_div <= 8; grid_div++) {
    for (int n = 0; n < this->blob_bottom_->num(); ++n) {
      const int grid_dim = default_grid_dim / grid_div;
      TypeParam* top_data_gpu = this->blob_top_->mutable_gpu_data();
      // NOLINT_NEXT_LINE(whitespace/operators)
      vol2col_gpu_kernel<TypeParam, 2><<<grid_dim, CAFFE_CUDA_NUM_THREADS>>>(
          num_kernels, bottom_data_gpu + this->blob_bottom_->offset(n),
          this->blob_bottom_->gpu_shape() + 1, this->blob_top_->gpu_shape() + 1,
          this->blob_kernel_shape_->gpu_data(), this->blob_pad_->gpu_data(),
          this->blob_stride_->gpu_data(),
          top_data_gpu + this->blob_top_->offset(n),1);
      CUDA_POST_KERNEL_CHECK;
    }

    // Compare results against CPU version
    for (int i = 0; i < this->blob_top_->count(); ++i) {
      TypeParam cpuval = top_data_cpu[i];
      TypeParam gpuval = this->blob_top_->cpu_data()[i];
      EXPECT_EQ(cpuval, gpuval);
      if (cpuval != gpuval) {
        break;
      }
    }
  }
}

}  // namespace caffe
