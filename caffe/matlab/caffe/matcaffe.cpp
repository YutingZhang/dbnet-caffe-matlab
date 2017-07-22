// Copyright 2014 BVLC and contributors.
//
// matcaffe.cpp provides a wrapper of the caffe::Net class as well as some
// caffe::Caffe functions so that one could easily call it from matlab.
// Note that for matlab, we will simply use float as the data type.

#define MAT_CAFFE_VERSION 20150124

#include <sstream>
#include <string>
#include <vector>
#include <list>
#include <map>

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include "mex.h"

#include "caffe/caffe.hpp"
#include "caffe/sgd_solvers.hpp"

#define MEX_ARGS int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs

// Log and throw a Mex error
inline void mex_error(const std::string &msg) {
  LOG(ERROR) << msg;
  mexErrMsgTxt(msg.c_str());
}

using namespace caffe;  // NOLINT(build/namespaces)

namespace caffe { typedef SGDSolver<float> solver_t; typedef Net<float> net_t; }

// Five things to be aware of:
//   caffe uses row-major order
//   matlab uses column-major order
//   caffe uses BGR color channel order
//   matlab uses RGB color channel order
//   images need to have the data mean subtracted
//
// Data coming in from matlab needs to be in the order
//   [width, height, channels, images]
// where width is the fastest dimension.
// Here is the rough matlab for putting image data into the correct
// format:
//   % convert from uint8 to single
//   im = single(im);
//   % reshape to a fixed size (e.g., 227x227)
//   im = imresize(im, [IMAGE_DIM IMAGE_DIM], 'bilinear');
//   % permute from RGB to BGR and subtract the data mean (already in BGR)
//   im = im(:,:,[3 2 1]) - data_mean;
//   % flip width and height to make width the fastest dimension
//   im = permute(im, [2 1 3]);
//
// If you have multiple images, cat them with cat(4, ...)
//
// The actual forward function. It takes in a cell array of 4-D arrays as
// input and outputs a cell array.


namespace caffe {

class SolverWrapper {
private:
	solver_t& solver_;
	string default_lr_policy_;
	float default_base_lr_;
public:
	SolverWrapper( solver_t& solver ) :
		solver_( solver ), default_lr_policy_( solver.param_.lr_policy() ),
		default_base_lr_( solver.param_.base_lr() ) {}

	void SetLearningRate2Default() {
		solver_.param_.set_lr_policy( default_lr_policy_ );
		solver_.param_.set_base_lr( default_base_lr_ );
	}
	void SetLearningRate2Fixed( float lr ) {
		solver_.param_.set_lr_policy( "fixed" );
		solver_.param_.set_base_lr( lr );
	}

	enum {
		START_ITER,
		CUR_ITER
	};
	void IncIter( int inc_amount = 1, int rel_pos = CUR_ITER ) {
		switch (rel_pos) {
		case START_ITER:
			solver_.iter_ = 0;
		case CUR_ITER:
			solver_.iter_ += inc_amount;
			break;
		default:
			LOG(FATAL) << "IncIter : Unrecognized rel_pos" << std::endl;
		}
	}
	void ApplyUpdate() { solver_.ApplyUpdate(); }
	void PreSolve() { solver_.PreSolve(); }
    virtual ~SolverWrapper() {
    	LOG(INFO) << "SolverWrapper is destroyed";
    }
protected:
	DISABLE_COPY_AND_ASSIGN(SolverWrapper);
};

class NetWrapper {
    net_t& net_;
public:
    NetWrapper( net_t& net ) : net_(net) {}
    void set_phase( caffe::Phase phase ) { // not officially supported, use it at your own risk
      net_.phase_ = phase;
      const vector<shared_ptr<Layer<float> > >& layers = net_.layers();
      for (size_t layer_id = 0; layer_id < layers.size(); ++layer_id) {
          // Inherit phase from net if unset.
          layers[layer_id]->phase_ = phase;
      }
    }
    void clear_diff() {
        for (int i = 0; i < net_.params().size(); ++i) {
          shared_ptr<Blob<float> > blob = net_.params()[i];
          switch (Caffe::mode()) {
          case Caffe::CPU:
            caffe_set(blob->count(), static_cast<float>(0),
                blob->mutable_cpu_diff());
            break;
          case Caffe::GPU:
    #ifndef CPU_ONLY
            caffe_gpu_set(blob->count(), static_cast<float>(0),
                blob->mutable_gpu_diff());
    #else
            NO_GPU;
    #endif
            break;
          }
        }
    }

    virtual ~NetWrapper() {
    	LOG(INFO) << "NetWrapper is destroyed";
    }
protected:
	DISABLE_COPY_AND_ASSIGN(NetWrapper);
};

}

static shared_ptr< NetWrapper > net_wrapper_;
static shared_ptr< SolverWrapper > solver_wrapper_;

// The pointer to the internal caffe::Net instance
static shared_ptr< net_t > net_;
static shared_ptr< solver_t > solver_;
static int init_key = -2;



static void do_save_net( const std::string& filename ) {
	NetParameter net_param;
	net_->ToProto(&net_param);
	WriteProtoToBinaryFile(net_param, filename.c_str());
}

static void do_set_layer_weights(
        const int layer_ids1, const mxArray* const layer_weights ) {
    const int layer_ids = layer_ids1 - 1; // 1-base to 0-base
    if (layer_ids<0) {
        mex_error("Wrong layer id (less than 0)");
    }
    
    const vector<shared_ptr<Layer<float> > >& layers = net_->layers();
    const vector<string>& layer_names = net_->layer_names();
    
    // find the target layer blobs
    vector<shared_ptr<Blob<float> > >* layer_blobs_ptr = NULL;
    std::string layer_name;
    int cur_layer_ids = 0;
    {
        string prev_layer_name = "";
        for (unsigned int i = 0; i < layers.size(); ++i) {
          vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
          if (layer_blobs.size() == 0) {
            continue;
          }
          if ( cur_layer_ids == layer_ids ) {
              layer_blobs_ptr = &layer_blobs;
              layer_name = layer_names[i];
              break;
          }
          if (layer_names[i] != prev_layer_name) {
            prev_layer_name = layer_names[i];
            cur_layer_ids++;
          }
        }
    }
    
    if (!layer_blobs_ptr) {
        mex_error("Wrong layer id (too large)");
    }
    
    vector<shared_ptr<Blob<float> > >& layer_blobs = *layer_blobs_ptr;
    
    // check the input
    {
        const size_t layer_weights_numel = mxGetNumberOfElements(layer_weights);
        if ( layer_weights_numel != layer_blobs.size() ) {
            mex_error("Wrong blob number");
        }
    }
    
    for ( size_t i = 0; i<layer_blobs.size(); ++i ) {
        const mxArray* const cur_weight = mxGetCell( layer_weights, static_cast<mwIndex>(i) );
        if ( !mxIsSingle(cur_weight) ) {
            mex_error("Weights must be SINGLE");
        }
        const mwSize  cur_weight_ndim = mxGetNumberOfDimensions( cur_weight );
        const mwSize* cur_weight_dims = mxGetDimensions( cur_weight );
        
        mwSize idims[4] = {1, 1, 1, 1};
        if (cur_weight_ndim<=4) {
            for (mwSize k=0;k<cur_weight_ndim;++k) 
                idims[k] = cur_weight_dims[k];
        } else {
            mex_error("Invalid weights dims");
        };
        
        mwSize dims[4] = {layer_blobs[i]->width(), layer_blobs[i]->height(),
             layer_blobs[i]->channels(), layer_blobs[i]->num()};
            
        for ( int j=0; j<4;++j ) {
            if ( idims[j] != dims[j] ) {
                LOG(ERROR) << "Wrong weights dims for Blob " << i 
                    << " in Layer " << layer_name;
                LOG(ERROR) << "Internal dims: " 
                    << dims[0] << ", "
                    << dims[1] << ", "
                    << dims[2] << ", "
                    << dims[3];
                LOG(ERROR) << "Input dims:    " 
                    << idims[0] << ", "
                    << idims[1] << ", "
                    << idims[2] << ", "
                    << idims[3];
                LOG(ERROR) << "Input ndims :  " << cur_weight_ndim;
                mex_error("Wrong weights dims");
            }
        }
    }
    
    // copy the content
    for ( size_t i = 0; i<layer_blobs.size(); ++i ) {
        const mxArray* const cur_weight = mxGetCell( layer_weights, static_cast<mwIndex>(i) );
        
		const float* weights_ptr = reinterpret_cast<float*>(mxGetPr( cur_weight ));

        //  mexPrintf("layer: %s (%d) blob: %d  %d: (%d, %d, %d) %d\n",
        //  layer_names[i].c_str(), i, j, layer_blobs[j]->num(),
        //  layer_blobs[j]->height(), layer_blobs[j]->width(),
        //  layer_blobs[j]->channels(), layer_blobs[j]->count());

        switch (Caffe::mode()) {
        case Caffe::CPU:
          caffe_copy(layer_blobs[i]->count(),weights_ptr,
        		  layer_blobs[i]->mutable_cpu_data());
          break;
        case Caffe::GPU:
          caffe_copy(layer_blobs[i]->count(),weights_ptr,
        		  layer_blobs[i]->mutable_gpu_data());
          break;
        default:
          LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
        }
        
    }

}

static mxArray* do_get_layer_names() {
  const vector<shared_ptr<Layer<float> > >& layers = net_->layers();
  const vector<string>& layer_names = net_->layer_names();

  std::list<std::string> layer_names_list;
  int num_layers = 0;
  {
    string prev_layer_name = "";
    for (unsigned int i = 0; i < layers.size(); ++i) {
      vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
      if (layer_blobs.size() == 0) {
        continue;
      }
      if (layer_names[i] != prev_layer_name) {
        prev_layer_name = layer_names[i];
        layer_names_list.push_back( prev_layer_name );
        num_layers++;
      }
    }
  }
  
  mxArray* l = mxCreateCellMatrix(num_layers, 1);
  {
      std::list<std::string>::iterator iter = layer_names_list.begin();
      for ( int i = 0; i<num_layers; ++i ) {
          mxSetCell( l, i, mxCreateString( iter->c_str() ) );
          ++iter;
      }
  }
  
  return l;
  
}

static mxArray* do_get_response_ids( const vector<Blob<float>*>& target_blobs ) {
    const vector<shared_ptr<Blob<float> > >& blobs = net_->blobs();

    typedef map<const Blob<float>*, size_t> blob_id_map_t;
    blob_id_map_t bim;
    for ( size_t i=0; i<blobs.size(); ++i )
        bim[blobs[i].get()] = i + 1;  // to 1-base

    vector<size_t> target_ids( target_blobs.size() );
    for ( size_t i=0; i<target_blobs.size(); ++i ) {
        target_ids[i] = bim[target_blobs[i]];  // if cannot find, give 0
    }

    mxArray* mx_out = mxCreateDoubleMatrix( target_blobs.size(), 1, mxREAL );
    double* target_ids_double = mxGetPr(mx_out);
    for ( size_t i=0; i<target_blobs.size(); ++i ) {
        target_ids_double[i] = static_cast<double>(target_ids[i]);
    }

    return mx_out;
}

static mxArray* do_get_input_response_ids() {
	return do_get_response_ids( net_->input_blobs() );
}

static mxArray* do_get_output_response_ids() {
	return do_get_response_ids( net_->output_blobs() );
}

static mxArray* do_get_response_info() {
	const vector<string>& blob_names = net_->blob_names();
	const vector<shared_ptr<Blob<float> > >& blobs = net_->blobs();
	size_t n = blob_names.size();

	mxArray* mx_out;
	{
		const mwSize dims[2] = {(mwSize)n, 1};
		const char* fnames[2] = {"name", "size"};
		mx_out = mxCreateStructArray(2, dims, 2, fnames);
	}

	for (unsigned int i = 0; i < n; ++i) {
		mxArray* mx_blob_name = mxCreateString(blob_names[i].c_str());
		mxArray* mx_blob_size = mxCreateDoubleMatrix(1,4,mxREAL);
		double* mx_blob_size_double = mxGetPr( mx_blob_size );
		mx_blob_size_double[0] = blobs[i]->width();
		mx_blob_size_double[1] = blobs[i]->height();
		mx_blob_size_double[2] = blobs[i]->channels();
		mx_blob_size_double[3] = blobs[i]->num();
    	mxSetField(mx_out, i, "name", mx_blob_name);
    	mxSetField(mx_out, i, "size",mx_blob_size);
	}
	return mx_out;
}

static void do_forward_no_output(const mxArray* const bottom) {
  const vector<Blob<float>*>& input_blobs = net_->input_blobs();
  if (static_cast<unsigned int>(mxGetDimensions(bottom)[0]) !=
      input_blobs.size()) {
    mex_error("Invalid input size");
  }
  for (unsigned int i = 0; i < input_blobs.size(); ++i) {
    const mxArray* const elem = mxGetCell(bottom, i);
    if (!mxIsSingle(elem)) {
      mex_error("MatCaffe require single-precision float point data");
    }
    if (mxGetNumberOfElements(elem) != input_blobs[i]->count()) {
      std::string error_msg;
      error_msg += "MatCaffe input size does not match the input size ";
      error_msg += "of the network";
      mex_error(error_msg);
    }

    const float* const data_ptr =
        reinterpret_cast<const float* const>(mxGetPr(elem));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(input_blobs[i]->count(), data_ptr,
          input_blobs[i]->mutable_cpu_data());
      break;
    case Caffe::GPU:
      caffe_copy(input_blobs[i]->count(), data_ptr,
          input_blobs[i]->mutable_gpu_data());
      break;
    default:
      mex_error("Unknown Caffe mode");
    }  // switch (Caffe::mode())
  }
}

static mxArray* do_response(const vector<Blob<float>*>& target_blobs) {
	size_t n = target_blobs.size();
	mxArray* mx_out = mxCreateCellMatrix(n, 1);
	for (unsigned int i = 0; i < n; ++i) {
	    // internally data is stored as (width, height, channels, num)
        // where width is the fastest dimension
        mwSize dims[4] = {target_blobs[i]->width(), target_blobs[i]->height(),
            target_blobs[i]->channels(), target_blobs[i]->num()};
        mxArray* mx_blob =  mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
        // OLD: output a vector instead of an array
 	    // mxArray* mx_blob =  mxCreateNumericMatrix(target_blobs[i]->count(), 1,
		//     mxSINGLE_CLASS, mxREAL);
		mxSetCell(mx_out, i, mx_blob);
		float* data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
		switch (Caffe::mode()) {
		case Caffe::CPU:
			caffe_copy(target_blobs[i]->count(),target_blobs[i]->cpu_data(),data_ptr);
			break;
		case Caffe::GPU:
			caffe_copy(target_blobs[i]->count(),target_blobs[i]->gpu_data(),data_ptr);
			break;
		default:
			LOG(FATAL)<< "Unknown Caffe mode.";
		}  // switch (Caffe::mode())
	}

	return mx_out;
}

static mxArray* do_response(const mxArray* const blob_ids) {

	const vector<shared_ptr<Blob<float> > >& blobs = net_->blobs();
	size_t blob_num = net_->blobs().size();
	size_t n = static_cast<size_t>(mxGetM(blob_ids) * mxGetN(blob_ids));
	double* ids_double = mxGetPr(blob_ids);
	vector<Blob<float>*> output_blobs(n);
	for ( size_t i=0; i<n; ++i ) {
		size_t ids_i  = static_cast<size_t>(ids_double[i]) - 1; // 1-base to 0-base
		if (ids_i>=blob_num) {
			mex_error("Wrong blob id");
		}
		output_blobs[i] = blobs[ids_i].get();
	}

	return do_response( output_blobs );

} // do_response

static mxArray* do_backward(const mxArray* const top_diff) {
  const vector<Blob<float>*>& output_blobs = net_->output_blobs();
  const vector<Blob<float>*>& input_blobs = net_->input_blobs();
  if (static_cast<unsigned int>(mxGetDimensions(top_diff)[0]) !=
      output_blobs.size()) {
    mex_error("Invalid input size");
  }
  // First, copy the output diff
  for (unsigned int i = 0; i < output_blobs.size(); ++i) {
    const mxArray* const elem = mxGetCell(top_diff, i);
    const float* const data_ptr =
        reinterpret_cast<const float* const>(mxGetPr(elem));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(output_blobs[i]->count(), data_ptr,
          output_blobs[i]->mutable_cpu_diff());
      break;
    case Caffe::GPU:
      caffe_copy(output_blobs[i]->count(), data_ptr,
          output_blobs[i]->mutable_gpu_diff());
      break;
    default:
        mex_error("Unknown Caffe mode");
    }  // switch (Caffe::mode())
  }
  // LOG(INFO) << "Start";
  net_->Backward();
  // LOG(INFO) << "End";
  mxArray* mx_out = mxCreateCellMatrix(input_blobs.size(), 1);
  for (unsigned int i = 0; i < input_blobs.size(); ++i) {
    // internally data is stored as (width, height, channels, num)
    // where width is the fastest dimension
    mwSize dims[4] = {input_blobs[i]->width(), input_blobs[i]->height(),
      input_blobs[i]->channels(), input_blobs[i]->num()};
    mxArray* mx_blob =  mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
    mxSetCell(mx_out, i, mx_blob);
    float* data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
    switch (Caffe::mode()) {
    case Caffe::CPU:
      caffe_copy(input_blobs[i]->count(), input_blobs[i]->cpu_diff(), data_ptr);
      break;
    case Caffe::GPU:
      caffe_copy(input_blobs[i]->count(), input_blobs[i]->gpu_diff(), data_ptr);
      break;
    default:
        mex_error("Unknown Caffe mode");
    }  // switch (Caffe::mode())
  }
  
  return mx_out;
}

static void do_set_layer_weights2(const mxArray* const layer_name,
    const mxArray* const mx_layer_weights) {
  const vector<shared_ptr<Layer<float> > >& layers = net_->layers();
  const vector<string>& layer_names = net_->layer_names();

  char* c_layer_names = mxArrayToString(layer_name);
  LOG(INFO) << "Looking for: " << c_layer_names;

  for (unsigned int i = 0; i < layers.size(); ++i) {

    if (strcmp(layer_names[i].c_str(),c_layer_names) == 0) {
      vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
      if (layer_blobs.size() == 0) {
        continue;
      }

      if ( static_cast<unsigned int>(mxGetDimensions(mx_layer_weights)[0]) 
              != layer_blobs.size() )
          mex_error( "Num of cells don't match layer_blobs.size" );
      LOG(INFO) << "layer_blobs.size() = " << layer_blobs.size();
      for (unsigned int j = 0; j < layer_blobs.size(); ++j) {
        // internally data is stored as (width, height, channels, num)
        // where width is the fastest dimension
        const mxArray* const elem = mxGetCell(mx_layer_weights, j);
        //mwSize dims[4] = {layer_blobs[j]->width(), layer_blobs[j]->height(),
        //    layer_blobs[j]->channels(), layer_blobs[j]->num()};

        if (layer_blobs[j]->count() != mxGetNumberOfElements(elem) )
            mex_error("Numel of weights don't match count of layer_blob");
        //const mwSize* dims_elem = mxGetDimensions(elem);

        const float* const data_ptr =
            reinterpret_cast<const float* const>(mxGetPr(elem));


        switch (Caffe::mode()) {
        case Caffe::CPU:
          memcpy(layer_blobs[j]->mutable_cpu_data(), data_ptr,
              sizeof(float) * layer_blobs[j]->count());
          break;
        case Caffe::GPU:
#ifndef CPU_ONLY
          cudaMemcpy(layer_blobs[j]->mutable_gpu_data(), data_ptr,
              sizeof(float) * layer_blobs[j]->count(), cudaMemcpyHostToDevice);
#endif
          break;
        default:
          LOG(FATAL) << "Unknown Caffe mode.";
        }
      }
    }
  }
}


static mxArray* do_get_all_data() {
  const vector<shared_ptr<Blob<float> > >& blobs = net_->blobs();
  const vector<string>& blob_names = net_->blob_names();

  // Step 1: prepare output array of structures
  mxArray* mx_all_data;
  {
    const mwSize num_blobs[1] = {(int)blobs.size()};
    const char* fnames[2] = {"name", "data"};
    mx_all_data = mxCreateStructArray(1, num_blobs, 2, fnames);
  }

  for (unsigned int i = 0; i < blobs.size(); ++i) {

    mwSize dims[4] = {blobs[i]->width(), blobs[i]->height(),
        blobs[i]->channels(), blobs[i]->num()};

    mxArray* mx_blob_data =
      mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);

    float* blob_data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob_data));

    switch (Caffe::mode()) {
    case Caffe::CPU:
      memcpy(blob_data_ptr, blobs[i]->cpu_data(),
          sizeof(float) * blobs[i]->count());
      break;
    case Caffe::GPU:
#ifndef CPU_ONLY
      CUDA_CHECK(cudaMemcpy(blob_data_ptr, blobs[i]->gpu_data(),
          sizeof(float) * blobs[i]->count(), cudaMemcpyDeviceToHost));
#endif
      break;
    default:
      LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
    }
    mxSetField(mx_all_data, i, "name",
        mxCreateString(blob_names[i].c_str()));
    mxSetField(mx_all_data, i, "data",mx_blob_data);
  }
  return mx_all_data;
}

static mxArray* do_get_all_diff() {
  const vector<shared_ptr<Blob<float> > >& blobs = net_->blobs();
  const vector<string>& blob_names = net_->blob_names();

  // Step 1: prepare output array of structures
  mxArray* mx_all_diff;
  {
    const mwSize num_blobs[1] = {(int)blobs.size()};
    const char* fnames[2] = {"name", "diff"};
    mx_all_diff = mxCreateStructArray(1, num_blobs, 2, fnames);
  }

  for (unsigned int i = 0; i < blobs.size(); ++i) {

    mwSize dims[4] = {blobs[i]->width(), blobs[i]->height(),
        blobs[i]->channels(), blobs[i]->num()};

    mxArray* mx_blob_data =
      mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);

    float* blob_data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob_data));

    switch (Caffe::mode()) {
    case Caffe::CPU:
      memcpy(blob_data_ptr, blobs[i]->cpu_diff(),
          sizeof(float) * blobs[i]->count());
      break;
    case Caffe::GPU:
#ifndef CPU_ONLY
      CUDA_CHECK(cudaMemcpy(blob_data_ptr, blobs[i]->gpu_diff(),
          sizeof(float) * blobs[i]->count(), cudaMemcpyDeviceToHost));
#endif
      break;
    default:
      LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
    }
    mxSetField(mx_all_diff, i, "name",
        mxCreateString(blob_names[i].c_str()));
    mxSetField(mx_all_diff, i, "diff",mx_blob_data);
  }
  return mx_all_diff;
}

static void do_clear_diff() {
    net_wrapper_->clear_diff();
}

static mxArray* do_forward(const mxArray* const bottom) {
  if (solver_wrapper_)
    do_clear_diff();
  do_forward_no_output( bottom );
  const vector<Blob<float>*>& output_blobs = net_->ForwardPrefilled();
  mxArray* mx_out = do_response( output_blobs );
  return mx_out;
}

static mxArray* do_get_weights() {
  const vector<shared_ptr<Layer<float> > >& layers = net_->layers();
  const vector<string>& layer_names = net_->layer_names();

  // Step 1: count the number of layers
  int num_layers = 0;
  {
    string prev_layer_name = "";
    for (unsigned int i = 0; i < layers.size(); ++i) {
      vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
      if (layer_blobs.size() == 0) {
        continue;
      }
      if (layer_names[i] != prev_layer_name) {
        prev_layer_name = layer_names[i];
        num_layers++;
      }
    }
  }

  // Step 2: prepare output array of structures
  mxArray* mx_layers;
  {
    const mwSize dims[2] = {num_layers, 1};
    const char* fnames[2] = {"weights", "layer_names"};
    mx_layers = mxCreateStructArray(2, dims, 2, fnames);
  }

  // Step 3: copy weights into output
  {
    string prev_layer_name = "";
    int mx_layer_index = 0;
    for (unsigned int i = 0; i < layers.size(); ++i) {
      vector<shared_ptr<Blob<float> > >& layer_blobs = layers[i]->blobs();
      if (layer_blobs.size() == 0) {
        continue;
      }

      mxArray* mx_layer_cells = NULL;
      if (layer_names[i] != prev_layer_name) {
        prev_layer_name = layer_names[i];
        const mwSize dims[2] = {static_cast<mwSize>(layer_blobs.size()), 1};
        mx_layer_cells = mxCreateCellArray(2, dims);
        mxSetField(mx_layers, mx_layer_index, "weights", mx_layer_cells);
        mxSetField(mx_layers, mx_layer_index, "layer_names",
            mxCreateString(layer_names[i].c_str()));
        mx_layer_index++;
      }

      for (unsigned int j = 0; j < layer_blobs.size(); ++j) {
        // internally data is stored as (width, height, channels, num)
        // where width is the fastest dimension
        mwSize dims[4] = {layer_blobs[j]->width(), layer_blobs[j]->height(),
            layer_blobs[j]->channels(), layer_blobs[j]->num()};
        mxArray* mx_weights = mxCreateNumericArray(4, dims, mxSINGLE_CLASS,
                                                   mxREAL);
        mxSetCell(mx_layer_cells, j, mx_weights);
        float* weights_ptr = reinterpret_cast<float*>(mxGetPr(mx_weights));

        //  mexPrintf("layer: %s (%d) blob: %d  %d: (%d, %d, %d) %d\n",
        //  layer_names[i].c_str(), i, j, layer_blobs[j]->num(),
        //  layer_blobs[j]->height(), layer_blobs[j]->width(),
        //  layer_blobs[j]->channels(), layer_blobs[j]->count());

        switch (Caffe::mode()) {
        case Caffe::CPU:
          caffe_copy(layer_blobs[j]->count(), layer_blobs[j]->cpu_data(),
              weights_ptr);
          break;
        case Caffe::GPU:
          caffe_copy(layer_blobs[j]->count(), layer_blobs[j]->gpu_data(),
              weights_ptr);
          break;
        default:
          mex_error("Unknown Caffe mode");
        }
      }
    }
  }

  return mx_layers;
}

static void get_weights(MEX_ARGS) {
  plhs[0] = do_get_weights();
}

static void set_mode_cpu(MEX_ARGS) {
  Caffe::set_mode(Caffe::CPU);
}

static void set_mode_gpu(MEX_ARGS) {
  Caffe::set_mode(Caffe::GPU);
}

static void set_phase_train(MEX_ARGS) {
  //LOG(WARNING) << "matcaffe: set_phase_* may be deprecated";
  if (net_) {
      net_wrapper_->set_phase(caffe::TRAIN);
  } else {
      mex_error( "No net is initialized" );
  }
}

static void set_phase_test(MEX_ARGS) {
  //LOG(WARNING) << "matcaffe: set_phase_* may be deprecated";
  if (net_) {
      net_wrapper_->set_phase(caffe::TEST);
  } else {
      mex_error( "No net is initialized" );
  }
}

static void set_phase_dummy(MEX_ARGS) {
  //LOG(WARNING) << "matcaffe: set_phase_* may be deprecated";
  if (net_) {
      net_wrapper_->set_phase(caffe::DUMMY);
  } else {
      mex_error( "No net is initialized" );
  }
}

static void set_device(MEX_ARGS) {
  if (nrhs != 1) {
    ostringstream error_msg;
    error_msg << "Expected 1 argument, got " << nrhs;
    mex_error(error_msg.str());
  }

  int device_id = static_cast<int>(mxGetScalar(prhs[0]));
  Caffe::SetDevice(device_id);
}

static void get_init_key(MEX_ARGS) {
  plhs[0] = mxCreateDoubleScalar(init_key);
}

static void init(MEX_ARGS) {
  if (nrhs != 3) {
    ostringstream error_msg;
    error_msg << "Expected 3 arguments, got " << nrhs;
    mex_error(error_msg.str());
  }

  char* param_file = mxArrayToString(prhs[0]);
  char* model_file = mxArrayToString(prhs[1]);
  char* phase_name = mxArrayToString(prhs[2]);

  Phase phase;
  if (strcmp(phase_name, "train") == 0) {
      phase = TRAIN;
  } else if (strcmp(phase_name, "test") == 0) {
	  phase = TEST;
  } else if (strcmp(phase_name, "dummy") == 0) {
      phase = DUMMY;
  } else {
    mex_error("Unknown phase.");
  }

  fs::path param_path( param_file );
  {
	  if (!fs::exists( param_path )) {
		  mex_error( "I cannot find the param_file" );
	  }

  }

  caffe::SolverParameter solver_param;
  bool is_solver = caffe::ReadProtoFromTextFile(param_file, &solver_param);
  LOG(INFO) << "Is a good solver file: " << is_solver;
  if ( is_solver ) { // try to load a solver first
	  string net_path_str = solver_param.net();
	  fs::path net_path( net_path_str );
	  if (net_path.empty()) {
		  mex_error( "net cannot be empty in the solver prototxt" );
	  }
	  if (!net_path.is_absolute()) {
		    net_path = param_path.parent_path() / net_path;
	  }
	  LOG(INFO) << "Train net absolute path : " << net_path;
	  if (!fs::exists(net_path)) {
		  mex_error( "I cannot find the net file" );
	  }
      net_path = fs::canonical(net_path);
      solver_param.set_net( net_path.native() );
      try {
        solver_t* tmp_solver = dynamic_cast< solver_t* >(caffe::SolverRegistry<float>::CreateSolver(solver_param));
	    solver_.reset( tmp_solver );
      } catch( ... ) {
          mex_error( "Cannot create solver." );
      }
	  net_ = solver_->net();
	  solver_wrapper_.reset( new SolverWrapper( *solver_ ) );
  } else { // load a network only
	  net_.reset(new Net<float>(string(param_file), phase));
  }
  net_wrapper_.reset( new NetWrapper( *net_ ) );

  if ( !string(model_file).empty() ) // if empty filename then, randomly initialized, or not intialized
    net_->CopyTrainedLayersFrom(string(model_file));

  mxFree(param_file);
  mxFree(model_file);
  mxFree(phase_name);

  // NOLINT_NEXT_LINE(runtime/threadsafe_fn)
  init_key = rand();
  if (nlhs == 1) {
    plhs[0] = mxCreateDoubleScalar(init_key);
  }
}

static void clear_diff(MEX_ARGS) {
  do_clear_diff();
}

static void reset(MEX_ARGS) {
  if (solver_) {
      solver_.reset();
      net_.reset();
      init_key = -2;
      LOG(INFO) << "Network reset, call init before use it again";
  } else {
    if (net_) {
        net_.reset();
        init_key = -2;
        LOG(INFO) << "Network reset, call init before use it again";
    }
  }
}

static void forward(MEX_ARGS) {
  if (nrhs != 1) {
    ostringstream error_msg;
    error_msg << "Expected 1 argument, got " << nrhs;
    mex_error(error_msg.str());
  }
  if (nlhs==1)
	  plhs[0] = do_forward(prhs[0]);
  else if (nlhs==0)
	  do_forward_no_output(prhs[0]);
  else {
	  mex_error("Wrong number of outputs");
  }
}

static void response(MEX_ARGS) {
  if (nrhs != 1) {
    mex_error("Wrong number of arguments");
  }
  if (!mxIsDouble( prhs[0] )) {
	mex_error("Arguments should be double");
  }
  if ( mxGetM( prhs[0] )>1. && mxGetN( prhs[0] ) ) {
		mex_error("The argument should be a vector");
  }
  plhs[0] = do_response(prhs[0]);
}

static void get_response_info(MEX_ARGS) {
	if (nrhs != 0) {
		mex_error("Wrong number of arguments");
	}
	plhs[0] = do_get_response_info();
}

static void get_output_response_ids(MEX_ARGS) {
	if (nrhs != 0) {
		mex_error("Wrong number of arguments");
	}
	plhs[0] = do_get_output_response_ids();
}

static void get_input_response_ids(MEX_ARGS) {
	if (nrhs != 0) {
		mex_error("Wrong number of arguments");
	}
	plhs[0] = do_get_input_response_ids();
}

static void backward(MEX_ARGS) {
  if (nrhs != 1) {
    ostringstream error_msg;
    error_msg << "Expected 1 argument, got " << nrhs;
    mex_error(error_msg.str());
  }

  plhs[0] = do_backward(prhs[0]);
}

static void is_initialized(MEX_ARGS) {
  if (!net_) {
    plhs[0] = mxCreateDoubleScalar(0);
  } else {
    plhs[0] = mxCreateDoubleScalar(1);
  }
}

static void get_layer_names(MEX_ARGS) {
  if (nrhs != 0) {
    mex_error("Wrong number of arguments");
  }
  plhs[0] = do_get_layer_names();
}

static void set_layer_weights(MEX_ARGS) {
  if (nrhs != 2) {
    mex_error("Wrong number of arguments");
  }
  
  if (!mxIsCell(prhs[1])) {
    mex_error("The second argument should be a cell array");
  }
  
  do_set_layer_weights( static_cast<int>( mxGetScalar(prhs[0]) ), prhs[1] );
}

static void save_net(MEX_ARGS) {
	if (nrhs != 1) {
		mex_error("Wrong number of arguments");
	}
	if ( !mxIsChar(prhs[0]) ) {
		mex_error("The argument should be string");
	}
	char* net_file_ = mxArrayToString(prhs[0]);
	std::string net_file(net_file_);
	mxFree(net_file_);
	do_save_net( net_file );
}

static void get_version(MEX_ARGS) {
	if (nrhs != 0) {
		mex_error("Wrong number of arguments");
	}
	
	plhs[0] = mxCreateDoubleScalar(MAT_CAFFE_VERSION);
}

static void update(MEX_ARGS) {
    solver_wrapper_->ApplyUpdate();
	//solver_wrapper_->ComputeUpdateValue();
    solver_wrapper_->IncIter();
	//solver_->net()->Update();
}

static void set_iter(MEX_ARGS) {
    if (nrhs != 1) {
        mexErrMsgTxt("Usage: caffe('set_iter', iter_num)");
        return;
    }
    int target_iter = static_cast<int>( mxGetScalar(prhs[0]) );
    solver_wrapper_->IncIter( target_iter, SolverWrapper::START_ITER );
}

static void set_lr(MEX_ARGS) {
    if (nrhs==0) {
        solver_wrapper_->SetLearningRate2Default();
    } else if (nrhs==1) {
        double target_lr = mxGetScalar(prhs[0]);
        solver_wrapper_->SetLearningRate2Fixed( target_lr );
    } else {
        mexErrMsgTxt("Usage: caffe('set_lr', iter_num)\n  or caffe('set_lr')");
        return;
    }
}

static void presolve(MEX_ARGS){
	if (nrhs != 0) {
        mexErrMsgTxt("Usage: caffe('presolve')");
        return;
	}
    LOG(INFO) << "No need to call presolve from now on";
	solver_wrapper_->PreSolve();
}


static void get_all_data(MEX_ARGS) {
  if (nrhs != 0) {
    mex_error("Wrong number of arguments");
  }
  plhs[0] = do_get_all_data();
}


static void get_all_diff(MEX_ARGS) {
  if (nrhs != 0) {
    mex_error("Wrong number of arguments");
  }
  plhs[0] = do_get_all_diff();
}

static void set_weights(MEX_ARGS) {
 if (nrhs != 1) {
     mex_error("Wrong number of arguments");
  }
  const mxArray* const mx_weights = prhs[0];
  if (!mxIsStruct(mx_weights)) {
     mexErrMsgTxt("Input needs to be struct");
  }
  int num_layers = mxGetNumberOfElements(mx_weights);
  // LOG(INFO) << "begin set layers with layer number: " << num_layers;
  for (int i = 0; i < num_layers; ++i) {
    const mxArray* layer_name= mxGetField(mx_weights,i,"layer_names");
    const mxArray* weights= mxGetField(mx_weights,i,"weights");
    do_set_layer_weights2(layer_name,weights);
  }
}


static void read_mean(MEX_ARGS) {
    if (nrhs != 1) {
        mexErrMsgTxt("Usage: caffe('read_mean', 'path_to_binary_mean_file'");
        return;
    }
    const string& mean_file = mxArrayToString(prhs[0]);
    Blob<float> data_mean;
    LOG(INFO) << "Loading mean file from: " << mean_file;
    BlobProto blob_proto;
    bool result = ReadProtoFromBinaryFile(mean_file.c_str(), &blob_proto);
    if (!result) {
        mexErrMsgTxt("Couldn't read the file");
        return;
    }
    data_mean.FromProto(blob_proto);
    mwSize dims[4] = {data_mean.width(), data_mean.height(),
                      data_mean.channels(), data_mean.num() };
    mxArray* mx_blob =  mxCreateNumericArray(4, dims, mxSINGLE_CLASS, mxREAL);
    float* data_ptr = reinterpret_cast<float*>(mxGetPr(mx_blob));
    caffe_copy(data_mean.count(), data_mean.cpu_data(), data_ptr);
    mexWarnMsgTxt("Remember that Caffe saves in [width, height, channels]"
                  " format and channels are also BGR!");
    plhs[0] = mx_blob;
}

/** -----------------------------------------------------------------
 ** Available commands.
 **/
struct handler_registry {
  string cmd;
  void (*func)(MEX_ARGS);
};

static handler_registry handlers[] = {
  // Public API functions
  { "save_net",           save_net },
  { "set_layer_weights",  set_layer_weights },
  { "get_layer_names",    get_layer_names },
  { "get_input_response_ids", get_input_response_ids },
  { "get_output_response_ids", get_output_response_ids },
  { "get_response_info",  get_response_info },
  { "get_response",       response        },
  { "set_weights",        set_weights     },
  { "response",           response        }, // same as get response
  { "presolve",           presolve        },
  { "update",             update          },
  { "set_iter",           set_iter        },
  { "set_lr",             set_lr          },
  { "get_version",        get_version     },
  { "get_all_diff",       get_all_diff    },
  { "get_all_data",       get_all_data    },
  { "set_phase_train",    set_phase_train },
  { "set_phase_test",     set_phase_test  },
  { "set_phase_dummy",    set_phase_dummy  },

  { "forward",            forward         },
  { "backward",           backward        },
  { "clear_diff",         clear_diff      },
  { "init",               init            },
  { "is_initialized",     is_initialized  },
  { "set_mode_cpu",       set_mode_cpu    },
  { "set_mode_gpu",       set_mode_gpu    },
  { "set_device",         set_device      },
  { "get_weights",        get_weights     },
  { "get_init_key",       get_init_key    },
  { "reset",              reset           },
  { "read_mean",          read_mean       },
  // The end.
  { "END",                NULL            },
};


/** -----------------------------------------------------------------
 ** matlab entry point: caffe(api_command, arg1, arg2, ...)
 **/
void mexFunction(MEX_ARGS) {
  mexLock();  // Avoid clearing the mex file.
  if (nrhs == 0) {
    mex_error("No API command given");
    return;
  }

  { // Handle input command
    char *cmd = mxArrayToString(prhs[0]);
    bool dispatched = false;
    // Dispatch to cmd handler
    for (int i = 0; handlers[i].func != NULL; i++) {
      if (handlers[i].cmd.compare(cmd) == 0) {
        handlers[i].func(nlhs, plhs, nrhs-1, prhs+1);
        dispatched = true;
        break;
      }
    }
    if (!dispatched) {
      ostringstream error_msg;
      error_msg << "Unknown command '" << cmd << "'";
      mex_error(error_msg.str());
    }
    mxFree(cmd);
  }
}
