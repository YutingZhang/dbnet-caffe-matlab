#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include <string>
#include <sstream>
#include <iomanip>
//#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string/replace.hpp>

using namespace std;
using namespace boost;

namespace caffe {

template <typename Dtype>
void Layer<Dtype>::snapshot_callback( const string& op, const vector<Blob<Dtype>*>& bottom,
	      const vector<Blob<Dtype>*>& top ) {
	LayerSnapshotParameter snapshot_param = this->layer_param_.snapshot();

	if ( op == "setup" ) {
		forward_count_  = -1;
		backward_count_ = -1;
	} else if ( op == "before-forward" ) {
		++forward_count_;
		backward_count_ = -1;
    } else if ( op == "after-forward" ) {
	} else if ( op == "before-backward" ) {
		++backward_count_;
    } else if ( op == "after-backward" ) {
	} else {
		LOG(FATAL) << "Unrecognized op for snapshot_callback";
    }

    string device = "-unknown";
    switch ( Caffe::mode() ) {
        case Caffe::CPU:
           device = "-CPU";
           break;
        case Caffe::GPU:
            device = "-GPU";
            if (this->cur_device){
                const int device_id = this->cur_device->device();
                device += to_string(device_id);
            }
           break;
    }

	if ( forward_count_>=0 && !(forward_count_%snapshot_param.interval()) ) {
		string iter_str;
		{
			std::ostringstream ss;
			ss << std::setw(10) << std::setfill('0') << forward_count_;
			iter_str = ss.str();
		}
		string fn = snapshot_param.output_prefix();
		boost::algorithm::replace_all(fn,"%%",layer_param_.name());
		boost::algorithm::replace_all(fn,"\%","%");
		fn += iter_str;
		if ( op == "before-forward" ) {
			if (snapshot_param.all()||snapshot_param.bottom_data()) {
				for ( size_t i=0; i<bottom.size(); ++i )
					ArrayToGZ( fn+"-bottom-data-"+to_string(i)+device+".blob.gz",
							bottom[i]->shape(), bottom[i]->cpu_data() );
			}
			if (snapshot_param.all()||snapshot_param.weights_data()) {
				for ( size_t i=0; i<blobs_.size(); ++i )
					ArrayToGZ( fn+"-weights-data-"+to_string(i)+device+".blob.gz",
							blobs_[i]->shape(), blobs_[i]->cpu_data() );
			}
		} else if ( op == "after-forward" ) {
			if (snapshot_param.all()||snapshot_param.top_data()) {
				for ( size_t i=0; i<top.size(); ++i )
					ArrayToGZ( fn+"-top-data-"+to_string(i)+device+".blob.gz",
							top[i]->shape(), top[i]->cpu_data() );
			}
		} else if ( op == "before-backward" ) {
			if (snapshot_param.all()||snapshot_param.top_diff()) {
				for ( size_t i=0; i<top.size(); ++i )
					ArrayToGZ( fn+"b"+to_string(backward_count_)+
							"-top-diff-"+to_string(i)+device+".blob.gz",
							top[i]->shape(), top[i]->cpu_diff() );
			}
		} else if ( op == "after-backward" ) {
			if (snapshot_param.all()||snapshot_param.bottom_diff()) {
				for ( size_t i=0; i<bottom.size(); ++i )
					ArrayToGZ( fn+"b"+to_string(backward_count_)+
							"-bottom-diff-"+to_string(i)+device+".blob.gz",
							bottom[i]->shape(), bottom[i]->cpu_diff() );
			}
			if (snapshot_param.all()||snapshot_param.weights_diff()) {
				for ( size_t i=0; i<blobs_.size(); ++i )
					ArrayToGZ( fn+"b"+to_string(backward_count_)+
							"-weights-diff-"+to_string(i)+device+".blob.gz",
							blobs_[i]->shape(), blobs_[i]->cpu_diff() );
			}
		} else {
			LOG(FATAL) << "Unrecognized op for snapshot_callback";
		}
	}
}

template void Layer<float>::snapshot_callback( const string& op, const vector<Blob<float>*>& bottom,
	      const vector<Blob<float>*>& top );

template void Layer<double>::snapshot_callback( const string& op, const vector<Blob<double>*>& bottom,
	      const vector<Blob<double>*>& top );


}
