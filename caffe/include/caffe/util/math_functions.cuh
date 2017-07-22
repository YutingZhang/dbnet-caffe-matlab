/*
 * math_functions.cuh
 *
 *  Created on: Sep 20, 2015
 *      Author: zhangyuting
 */

#ifndef MATH_FUNCTIONS_CUH_
#define MATH_FUNCTIONS_CUH_

#include "caffe/common.hpp"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

namespace caffe {

template <class DeviceMode, typename T>
struct thrust_vector { };

template<typename T>
struct thrust_vector<mCPU,T> {
	typedef thrust::host_vector<T> type;
};

template<typename T>
struct thrust_vector<mGPU,T> {
	typedef thrust::device_vector<T> type;
};

// convert a linear index to a non_channel index
template <typename T>
struct linear_index_to_nonCh_index : public thrust::unary_function<T,T>
{
protected:
  T I_; // inner number
  T O_; // outer number
  T C_;
public:
  __host__ __device__
  linear_index_to_nonCh_index(T I, T O, T C) : I_(I), O_(O), C_(C) {}

  __host__ __device__
  T operator()(T i)
  {
    return i/(I_*C_)*I_+i%I_;
  }

};

}

#endif /* MATH_FUNCTIONS_CUH_ */
