#include "caffe/filler.hpp"
#include <cmath>

namespace caffe {

template <typename Dtype>
void FilterFiller_Sobel<Dtype>::Fill(Blob<Dtype>* blob) {
	int numAxis = blob->num_axes();
	int outCh = blob->shape(0), inCh = blob->shape(1);
    CHECK_EQ(inCh, 1) << "Only support single input channel";
    CHECK_GE(numAxis-2,outCh) << "filter axis should be equal (or greater than output channels)";
    for (int i=2; i<numAxis-outCh; ++i)
    	CHECK_GE(blob->shape(i),1) << "axix " << i << " should be singleton in this setting";
    for (int i=numAxis-outCh; i<numAxis; ++i)
    	CHECK_GE(blob->shape(i),3) << "axix " << i << " should be 3-dim in this setting";

    Dtype p[3] = {Dtype(-1),Dtype(0),Dtype(1)}, h[3] = {Dtype(1),Dtype(2),Dtype(1)}; // for image conv, no flipping for h is needed
    Dtype* const data = blob->mutable_cpu_data();
    Dtype* f = data;
    int flatten_size = static_cast<int>( pow(Dtype(3),Dtype(outCh)) );
    for (int c = 0; c<outCh; ++c) {
    	for (int k = 0; k<flatten_size; ++k) {
    		int t = k;
    		Dtype v = Dtype(1.);
			for (int j=0; j<outCh; ++j) {
				int r = t % 3;
				v *= h[r];
				if (j==c) v *= p[r];
				t /= 3;
			}
			*(f++) = v;
    	}
    }
    /* for (int i=0; i<flatten_size*outCh; ++i) 
        LOG(INFO) << "FILTER: SOBEL: id: " << i << " : " << data[i]; */
    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
}

INSTANTIATE_CLASS( FilterFiller_Sobel );


template <typename Dtype>
void FilterFiller_Gaussian<Dtype>::Fill(Blob<Dtype>* blob) {
	const int numAxis = blob->num_axes();
	const int outCh = blob->shape(0), inCh = blob->shape(1);
    CHECK_EQ(inCh, 1) << "Only support single input channel";
    CHECK_EQ(outCh,1) << "Only support single output channel";
    const int filterAxis = numAxis-2;

    CHECK_GT(filterAxis,0) << "Filter axis should be greater than 1";

    std::vector<Dtype> filter_dims(filterAxis);
    int total_dim = 1;
    for (int i=0; i<filterAxis; ++i) {
    	filter_dims[i] = blob->shape(i+2);
    	CHECK_GT(filter_dims[i],0) << "dimension " << i+2 << " should be greater than zero";
    	total_dim *= filter_dims[i];
    }

    const Dtype defaultStd = Dtype(1);
    std::vector<Dtype> std_data(filterAxis, defaultStd);
    int stds_size = this->filler_param_.stds_size();
    if (stds_size==0) {
    	/* for (int i=0; i<filterAxis; ++i)
    		std_data[i] = defaultStd; */
    } else if (stds_size==1) {
    	for (int i=0; i<filterAxis; ++i)
    		std_data[i] = this->filler_param_.stds(0);
    } else {
    	CHECK_EQ( stds_size, filterAxis ) << "stds dimension mismatch with the filters";
    	for (int i=0; i<filterAxis; ++i)
    		std_data[i] = this->filler_param_.stds(i);
    }

    Dtype* data = blob->mutable_cpu_data(),* data_end = data + total_dim;
    *(data_end-1) = Dtype(1.);
    int left_length = 1;
    for (int i = filterAxis-1; i>=0; --i) {	// last dim goes first
    	const int right_length = filter_dims[i], block_dim = left_length*right_length;
    	Dtype* const left_ptr = data_end-left_length;
        Dtype* const block_ptr = data_end-block_dim;
    	Dtype* f = block_ptr;
    	long double x = -(long double)(right_length)/2.;
    	const long double a0 = std::erf( x ), aN = std::erf( -x );
    	const long double a_norm = aN-a0;
    	long double a1 = a0;
        /*LOG(INFO) << "FILTER: GAU: " << i << " / " << filterAxis << " : dim " << filter_dims[i]
            << " x: " << x << " a1: " << a1 << " aN: " << aN 
            << " left: " << left_length << " right: " << right_length; */
    	for ( int j = 0; j<right_length; ++j ) {
    		x += (long double)(1.);
    		long double a2 = std::erf( x ), v = (a2-a1)/a_norm;
            //LOG(INFO) << "a2: " << a2 << " v: " << v;
    		for ( int k = 0; k<left_length; ++k ) {
                long double new_v = (long double)(left_ptr[k]) * v;
                //LOG(INFO) << "new_v: " << new_v << " " << Dtype(new_v);
    			*(f++) = (Dtype)(new_v);
    		}
    		a1 = a2;
    	}
    	left_length = block_dim;
    }
    /* for (int i = 0; i<total_dim; ++i)
        LOG(INFO) << "FILTER: GAU: id: " << i << " : " << data[i]; */

    CHECK_EQ(this->filler_param_.sparse(), -1)
         << "Sparsity not supported by this Filler.";
}

INSTANTIATE_CLASS( FilterFiller_Gaussian );

}
