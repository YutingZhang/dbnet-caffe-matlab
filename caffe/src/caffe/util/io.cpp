#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <stdint.h>
#include <sstream>
#include <iomanip>

#include <boost/filesystem.hpp>
#include <cmath>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#include "caffe/util/zeta/utils.hpp"

#include <fstream>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/device/file.hpp>
#include <boost/iostreams/device/file_descriptor.hpp>
#include <boost/iostreams/filtering_stream.hpp>

const int kProtoReadBytesLimit = INT_MAX;  // Max size of 2 GB minus 1 byte.

namespace caffe {

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

bool ReadProtoFromTextFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  FileInputStream* input = new FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  return success;
}

void WriteProtoToTextFile(const Message& proto, const char* filename) {
  int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  FileOutputStream* output = new FileOutputStream(fd);
  CHECK(google::protobuf::TextFormat::Print(proto, output));
  delete output;
  close(fd);
}

bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
  coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 1024*1024*1024); // 536870912

  bool success = proto->ParseFromCodedStream(coded_input);

  delete coded_input;
  delete raw_input;
  close(fd);
  return success;
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename) {
  fstream output(filename, ios::out | ios::trunc | ios::binary);
  CHECK(proto.SerializeToOstream(&output));
}

#ifdef USE_OPENCV
cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color) {
  cv::Mat cv_img;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return cv_img_origin;
  }
  if (height > 0 && width > 0) {
    cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
  } else {
    cv_img = cv_img_origin;
  }
  return cv_img;
}

cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width) {
  return ReadImageToCVMat(filename, height, width, true);
}

cv::Mat ReadImageToCVMat(const string& filename,
    const bool is_color) {
  return ReadImageToCVMat(filename, 0, 0, is_color);
}

cv::Mat ReadImageToCVMat(const string& filename) {
  return ReadImageToCVMat(filename, 0, 0, true);
}

// Do the file extension and encoding match?
static bool matchExt(const std::string & fn,
                     std::string en) {
  size_t p = fn.rfind('.');
  std::string ext = p != fn.npos ? fn.substr(p) : fn;
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  std::transform(en.begin(), en.end(), en.begin(), ::tolower);
  if ( ext == en )
    return true;
  if ( en == "jpg" && ext == "jpeg" )
    return true;
  return false;
}

bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color,
    const std::string & encoding, Datum* datum) {
  cv::Mat cv_img = ReadImageToCVMat(filename, height, width, is_color);
  if (cv_img.data) {
    if (encoding.size()) {
      if ( (cv_img.channels() == 3) == is_color && !height && !width &&
          matchExt(filename, encoding) )
        return ReadFileToDatum(filename, label, datum);
      std::vector<uchar> buf;
      cv::imencode("."+encoding, cv_img, buf);
      datum->set_data(std::string(reinterpret_cast<char*>(&buf[0]),
                      buf.size()));
      datum->set_label(label);
      datum->set_encoded(true);
      return true;
    }
    CVMatToDatum(cv_img, datum);
    datum->set_label(label);
    return true;
  } else {
    return false;
  }
}
#endif  // USE_OPENCV

bool ReadFileToDatum(const string& filename, const int label,
    Datum* datum) {
  std::streampos size;

  fstream file(filename.c_str(), ios::in|ios::binary|ios::ate);
  if (file.is_open()) {
    size = file.tellg();
    std::string buffer(size, ' ');
    file.seekg(0, ios::beg);
    file.read(&buffer[0], size);
    file.close();
    datum->set_data(buffer);
    datum->set_label(label);
    datum->set_encoded(true);
    return true;
  } else {
    return false;
  }
}

#ifdef USE_OPENCV
cv::Mat DecodeDatumToCVMatNative(const Datum& datum) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  cv_img = cv::imdecode(vec_data, -1);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}
cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv_img = cv::imdecode(vec_data, cv_read_flag);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}

// If Datum is encoded will decoded using DecodeDatumToCVMat and CVMatToDatum
// If Datum is not encoded will do nothing
bool DecodeDatumNative(Datum* datum) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMatNative((*datum));
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}
bool DecodeDatum(Datum* datum, bool is_color) {
  if (datum->encoded()) {
    cv::Mat cv_img = DecodeDatumToCVMat((*datum), is_color);
    CVMatToDatum(cv_img, datum);
    return true;
  } else {
    return false;
  }
}

void CVMatToDatum(const cv::Mat& cv_img, Datum* datum) {
  CHECK(cv_img.depth() == CV_8U) << "Image data type must be unsigned byte";
  datum->set_channels(cv_img.channels());
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->clear_data();
  datum->clear_float_data();
  datum->set_encoded(false);
  int datum_channels = datum->channels();
  int datum_height = datum->height();
  int datum_width = datum->width();
  int datum_size = datum_channels * datum_height * datum_width;
  std::string buffer(datum_size, ' ');
  for (int h = 0; h < datum_height; ++h) {
    const uchar* ptr = cv_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < datum_width; ++w) {
      for (int c = 0; c < datum_channels; ++c) {
        int datum_index = (c * datum_height + h) * datum_width + w;
        buffer[datum_index] = static_cast<char>(ptr[img_index++]);
      }
    }
  }
  datum->set_data(buffer);
}

cv::Mat ReadImageToCVMat(const string& filename,
                         const int height, const int width, const bool is_color,
                         int* img_height, int* img_width) {
  cv::Mat cv_img;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return cv_img_origin;
  }

  if (height > 0 && width > 0) {
    cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
  } else {
    cv_img = cv_img_origin;
  }

  if (img_height != NULL) {
    *img_height = cv_img.rows;
  }
  if (img_width != NULL) {
    *img_width = cv_img.cols;
  }

  return cv_img;
}

cv::Mat ReadImageToCVMatNearest(const string& filename,
                         const int height, const int width, const bool is_color,
                         int* img_height, int* img_width) {
  cv::Mat cv_img;
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
  if (!cv_img_origin.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return cv_img_origin;
  }

  if (height > 0 && width > 0) {
    cv::resize(cv_img_origin, cv_img, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
  } else {
    cv_img = cv_img_origin;
  }

  if (img_height != NULL) {
    *img_height = cv_img.rows;
  }
  if (img_width != NULL) {
    *img_width = cv_img.cols;
  }

  return cv_img;
}


/* -------------- CUSTOM -------------- */

template <>
int SaveArrayAsImages<unsigned char>( int W, int H, int nCh, int num, const unsigned char* data,
		const std::string& file_prefix, int start_idx, size_t padding_zeros, const bool force_slice_channel ) {
	cv::Mat cv_img;
    int nChImg = 1, nSlice = 1;
	if (!force_slice_channel && nCh==3) {
		nChImg = 3;
		cv_img = cv::Mat(H,W,CV_8UC3);
	} else if (nCh>0) {
		nSlice = nCh;
		cv_img = cv::Mat(H,W,CV_8UC1);
    } else {
		LOG(ERROR) << "Must be positive number of channels";
	}

	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);

	size_t slice_padding_zeros = static_cast<size_t>(std::ceil(std::log(double(nSlice+1))));

	for ( int k = 0; k<num; ++k ) {
		std::string file_idx_str;
		{
			std::ostringstream ss;
			ss << std::setw(padding_zeros) << std::setfill('0') << (k+start_idx);
			file_idx_str = ss.str();
		}
		for ( int j = 0; j<nSlice; ++j ) {
			const unsigned char* cur_data = data + H*W*(nCh*k+j);
			for (int h = 0; h < H; ++h) {
				uchar* ptr = cv_img.ptr<uchar>(h);
				int img_index = 0;
				for (int w = 0; w < W; ++w) {
					for (int c = 0; c < nChImg; ++c) {
						int datum_index = (c * H + h) * W + w;
						ptr[img_index++] = static_cast<uchar>(cur_data[datum_index]);
					}
				}
			}
			std::string slice_idx_str;
			if (nSlice>1) {
				std::ostringstream ss;
				ss << "-" << std::setw(slice_padding_zeros) << std::setfill('0') << j;
				slice_idx_str = ss.str();
			}
			const std::string img_filename = file_prefix + file_idx_str + slice_idx_str + ".png";

			if (!k) create_parent_dir(img_filename);

			cv::imwrite( img_filename, cv_img, compression_params );
		}
	}
	return start_idx+num;
}

template <>
int SaveArrayAsImages<float>( int w, int h, int nCh, int num, const float* data,
		const std::string& file_prefix, int start_idx, size_t padding_zeros, const bool force_slice_channel ) {
	std::vector<unsigned char> data_c(w*h*nCh*num);
	for (size_t i=0; i<data_c.size(); ++i) {
        float a = std::floor(data[i]*256.f);
        a = std::min(255.f,std::max(0.f,a));
		data_c[i] = static_cast<unsigned char>(a);
    }
	return SaveArrayAsImages( w, h, nCh, num, &(data_c[0]), file_prefix, start_idx, padding_zeros, force_slice_channel );
}

template <>
int SaveArrayAsImages<double>( int w, int h, int nCh, int num, const double* data,
		const std::string& file_prefix, int start_idx, size_t padding_zeros, const bool force_slice_channel ) {
	std::vector<unsigned char> data_c(w*h*nCh*num);
	for (size_t i=0; i<data_c.size(); ++i) {
        double a = std::floor(data[i]*256.);
        a = std::min(255.,std::max(0.,a));
		data_c[i] = static_cast<unsigned char>(a);
    }
	return SaveArrayAsImages( w, h, nCh, num, &(data_c[0]), file_prefix, start_idx, padding_zeros, force_slice_channel );
}

template <>
int SaveArrayAsImages<signed char>( int w, int h, int nCh, int num, const signed char* data,
		const std::string& file_prefix, int start_idx, size_t padding_zeros, const bool force_slice_channel ) {
	const unsigned char* data1 = static_cast<const unsigned char*>(
            static_cast<const void*>(data));
	return SaveArrayAsImages( w, h, nCh, num, data1, file_prefix, start_idx, padding_zeros, force_slice_channel );
}

#endif  // USE_OPENCV

template <typename Dtype>
void ArrayToGZ_Internal ( const std::string& fn, const std::vector<size_t>& dims, const Dtype* data ) {
    using namespace boost::iostreams;
    using namespace std;
    const string type_str = zeta::matlab_type<Dtype>::name();

	size_t s = 1;
	if (dims.empty()) s=0;
	else {
		for ( size_t i = 0; i<dims.size(); ++i )
			s*=dims[i];
	}

	std::vector<uint64_t> header;
	header.push_back(dims.size());
	for ( size_t i=0; i<dims.size(); ++i )
		header.push_back(dims[i]);
    header.push_back(type_str.length());

	create_parent_dir( fn );

    filtering_ostream out;
	out.push(gzip_compressor(1)); // compression level = 1 for speed
	out.push(file_sink(fn, ios_base::out | ios_base::binary));
	{
		const void* a = static_cast<const void*>(header.data());
		size_t csize = header.size()*sizeof(uint64_t);
		out.write( static_cast<const char*>(a), csize );
	}
	out.write( type_str.c_str(), type_str.length() );
	out.write( static_cast<const char*>( static_cast<const void*>(data) ), s*sizeof(Dtype) );

}

template void ArrayToGZ_Internal<float> ( const std::string& fn, const std::vector<size_t>& dims, const float* data );
template void ArrayToGZ_Internal<double> ( const std::string& fn, const std::vector<size_t>& dims, const double* data );

void create_dir( const std::string& path, bool no_throw ) {
	boost::filesystem::path p(path);
    if (no_throw) {
        try { boost::filesystem::create_directories(p); }
        catch (...) {}
    } else {
	    boost::filesystem::create_directories(p);
    }
}

void create_parent_dir( const std::string& path, bool no_throw ) {
	boost::filesystem::path p(path);
    if (no_throw) {
        try { boost::filesystem::create_directories(p.parent_path()); }
        catch (...) {}
    } else {
	    boost::filesystem::create_directories(p.parent_path());
    }
}

bool file_exists( const std::string& path ) {
    std::ifstream infile(path);
    return infile.good();
}

}  // namespace caffe
