/*
 * fake_leveldb.cpp
 *
 *  Created on: Dec 29, 2013
 *      Author: zhangyuting
 */

#ifdef USE_OPENCV

#include "caffe/util/io.hpp"
#include "caffe/util/fake_leveldb.hpp"
#include "caffe/util/zeta/scheduler.hpp"
#include "caffe/util/zeta/misc_data.hpp"

#include <string>
#include <list>
#include <vector>
#include <fstream>
#include <iostream>
#include <map>
#include <cstring>
#include <algorithm>
#include <random>
#include <limits>

#include <opencv2/opencv.hpp>

#include <boost/thread.hpp>

#include "caffe/util/rng.hpp"
#include <boost/random/uniform_real.hpp>


using namespace std;

namespace caffe {


struct _fake_leveldb_iterator_inner {
    std::random_device rand_d;
    std::mt19937 rand_g;

    _fake_leveldb_iterator_inner() : rand_g( rand_d() ) {}
};

_fake_leveldb_iterator::_fake_leveldb_iterator() :
		a_( new _fake_leveldb_iterator_inner ) {}

// utils

int __translate_colortype2cv( int color_type ) {
	int cv_color_flag;
	switch( color_type ) {
	case fake_leveldb::BGR_COLOR:
		cv_color_flag = CV_LOAD_IMAGE_COLOR;
		break;
	case fake_leveldb::GRAY_SCALE:
		cv_color_flag = CV_LOAD_IMAGE_GRAYSCALE;
		break;
	case fake_leveldb::AS_IS_COLOR:
	default:
		cv_color_flag = CV_LOAD_IMAGE_UNCHANGED;
	}
	return cv_color_flag;
}

void im_resize( const cv::Mat& I0, cv::Mat& J, size2d canonical_size ) {
	cv::Mat I;
	float comp_scale = 1.f;
	if ( canonical_size.rot_max>0.f ) {
		// rand rotation
		float rng_a = caffe::rand_threadsafe();
		if ( canonical_size.rot_level>0 ) {
			int ss = signbit(rng_a);
			rng_a = std::abs(rng_a) * ((float)canonical_size.rot_level+0.5f - std::numeric_limits<float>::epsilon() ) + 0.5f;
			rng_a = rng_a / ((float)canonical_size.rot_level+1.f) * (float)ss; // angle will count twice
		}

		float a = rng_a*canonical_size.rot_max*2.-canonical_size.rot_max;
		// apply rotation

	    cv::Point2f pt(I0.cols/2., I0.rows/2.);
	    cv:: Mat r = cv::getRotationMatrix2D(pt, a, 1.0);
	    vector<cv::Point> points;
	    points.push_back( cv::Point(0,0) );
	    points.push_back( cv::Point(0,I0.rows) );
	    points.push_back( cv::Point(I0.cols, 0));
	    points.push_back( cv::Point(I0.cols, I0.rows) );
	    cv::transform( points, points, r );

	    float pleft(0), pright(I0.cols), ptop(0), pbottom(I0.rows);
	    for ( int i = 0; i<4; ++i ) {
	        float x = points[i].x, y = points[i].y;
	        if ( x <= pt.x ) {
	            if (pleft<x) pleft = x;
	        } else {
	            if (pright>x) pright = x;
	        }
	        if ( y <= pt.y ) {
	            if (ptop<y) ptop = y;
	        } else {
	            if (pbottom>y) pbottom = y;
	        }
	    }

	    cv::warpAffine(I0, I, r, I0.size(), CV_INTER_CUBIC+CV_WARP_FILL_OUTLIERS);
	    cv::getRectSubPix(I, cv::Size(pright-pleft,pbottom-ptop), pt, I);
	    comp_scale = (float)I.cols/(float)I0.cols;

	} else {
		I = I0;
	}
	if ( canonical_size.s>0 ) {
		cv::Size siz = I.size();
		double sw = double(canonical_size.s)/double(siz.width);
		double sh = double(canonical_size.s)/double(siz.height);
		double sc = std::max(sw,sh);
		double sc_min = std::max( double(canonical_size.s_min)/double(siz.width),
				double(canonical_size.s_min)/double(siz.height) );

		if (canonical_size.s_max>canonical_size.s) {
			float rand_enlarge_ratio = (float)canonical_size.s_max/(float)canonical_size.s;
			float rng_sc = caffe::rand_threadsafe();
			if ( canonical_size.s_level>0 ) {
				rng_sc = std::max( .0f, std::floor( rng_sc * ((float)canonical_size.s_level+1) - std::numeric_limits<float>::epsilon() ) /
						((float)canonical_size.s_level+1) );
			}
			float enlarge_s = std::pow( rand_enlarge_ratio, rng_sc );
			sc *= enlarge_s;
		}
		sc *= comp_scale;
		sc = std::max(sc,sc_min);

		cv::Size new_siz= cv::Size(
				(int)std::floor(siz.width*sc+0.5f),
				(int)std::floor(siz.height*sc+0.5f));
		//LOG(INFO) << "Resize (SHORTER) to " << new_siz.width << " x " << new_siz.height;
		cv::resize( I, J, new_siz, 0,0, CV_INTER_CUBIC );
	} else if ( canonical_size.w>0 && canonical_size.h>0 ) {
		// LOG(INFO) << "Resize to " << canonical_size.w << " x " << canonical_size.h;
		cv::resize( I, J, cv::Size(canonical_size.w,canonical_size.h), 0, 0, CV_INTER_CUBIC );
		// LOG(INFO) << "Im type : " << I.type() << " vs " << J.type();
	} else {
		CHECK( canonical_size.w<=0 && canonical_size.h<=0 ) << "Cannot specify only one of width and height";
		J = I;
	}
}


// --------------------------------------------- fake_leveldb_imagedb

fake_leveldb_imagedb::Iterator* fake_leveldb_imagedb::NewIterator(
		size_t max_open_files, size2d canonical_size, int color_type ) {
    return new Iterator( *root_dir_, *lines_, max_open_files, canonical_size, color_type );
}

int fake_leveldb_imagedb::OpenReadOnly( const string& database_path ) {

	string root_dir;
	list< labeled_data_t > lines;

	ifstream index_file( (database_path + "/index.txt").c_str() );
	if ( index_file.fail() ) {	// txt + image database, data are not consolidated

		string list_path;
		{
			ifstream db_in( database_path.c_str() );
			if ( db_in.fail() )
				return 1;
			if ( !( getline( db_in, root_dir ) && getline( db_in, list_path ) ) )
				return 2;
		}
		if ( !root_dir.empty() && root_dir[root_dir.length()-1]!='/' )
			root_dir += '/';
		if ( list_path.empty() )
			return 3;
		if ( list_path[0] != '/' ) {
			size_t p = database_path.rfind( '/' );
			if ( p != string::npos )
				list_path = string( database_path, 0, p+1 ) + list_path;
		}

		{
			ifstream infile( list_path.c_str() );
			if (infile.fail())
				return 11;
			string filename;
			int label;
			while (infile >> filename >> label) {
				lines.push_back(std::make_pair(filename, label));
			}
		}

	} else {

		root_dir = database_path;
		if ( !root_dir.empty() && root_dir[root_dir.length()-1]!='/' )
			root_dir += '/';
		string filename;
		size_t start_loc, length;
		int label;
		while (index_file >> filename >> start_loc >> length >> label) {
			lines.push_back(std::make_pair(
					fake_leveldb_imagedb::single_fileinfo
					(filename,start_loc,length), label));
		}

	}

	root_dir_.reset( new string(root_dir) );
	lines_.reset( new labeled_data_vec_t(lines.begin(), lines.end()) );


	return 0;
}


typedef zeta::raw_data_block compressed_image;

typedef zeta::restaurant_scheduler< compressed_image, cv::Mat > image_loader_scheduler;
typedef zeta::vec_block_data block_data;

class image_loader_worker : public image_loader_scheduler::worker {
	size2d canonical_size;
	int color_type;
public:
	image_loader_worker() : canonical_size(0), color_type(fake_leveldb::BGR_COLOR) {}
	explicit image_loader_worker( size2d canonical_size, int color_type ) :
		canonical_size(canonical_size), color_type(color_type) {}
	
	virtual cv::Mat operator () ( compressed_image c ) {
		block_data data(c.length);
		std::memcpy(&(data[0]),c.raw_data,c.length);
		cv::Mat I = cv::imdecode( data, __translate_colortype2cv(color_type) );
		cv::Mat J;
		im_resize( I, J, canonical_size );
		return J;
	}
	virtual image_loader_worker* clone() const {
		return new image_loader_worker( canonical_size, color_type );
	}
};

struct tdat_info {
	std::string filename;
	std::vector<fake_leveldb_imagedb::single_file_range> subfile_ranges;
	image_loader_scheduler* image_s;
};

struct tdat_jobs {
	block_data data;
	vector<image_loader_scheduler::key_t> keys;
	vector<bool> is_first_occurrence;
};

typedef zeta::restaurant_scheduler< boost::shared_ptr<tdat_info>,
		boost::shared_ptr<tdat_jobs> > tdat_loader_scheduler;

class tdat_loader_worker : public tdat_loader_scheduler::worker {
	virtual boost::shared_ptr<tdat_jobs> operator () (
			boost::shared_ptr<tdat_info> ti ) {

		boost::shared_ptr<tdat_jobs> tj( new tdat_jobs );

		size_t file_size;
		{
		    static boost::mutex io_race_mutex;
		    boost::mutex::scoped_lock io_race_lock(io_race_mutex);

			ifstream in( ti->filename.c_str(), std::ios_base::binary );
			in.seekg( 0, std::ios_base::end );
			file_size = in.tellg();
			in.seekg( 0, std::ios_base::beg );
			tj->data.resize( file_size );
			in.read( &(tj->data[0]), file_size );
		}


		std::vector<fake_leveldb_imagedb::single_file_range>& r = ti->subfile_ranges;
		if ( r.size() == 1 && r[0].start_loc == 0 )
			r[0].length = std::min( r[0].length, file_size );

		std::map<compressed_image,key_t> existing_ci;
		tj->keys.resize( r.size() );
		tj->is_first_occurrence.resize( r.size(), false );
		for ( size_t i=0; i<r.size(); ++i ) {
			compressed_image c;
			c.length = r[i].length;
			c.raw_data = &(tj->data[0])+r[i].start_loc;

			std::map<compressed_image,key_t>::iterator iter = existing_ci.find( c );
			if ( iter == existing_ci.end() ) {
				tj->keys[i] = ti->image_s->claim( c );
				tj->is_first_occurrence[i] = true;
				existing_ci[c] = tj->keys[i];
			} else {
				// allow image be loaded many times without occupying additional buffers
				tj->keys[i] = iter->second;
			}
		}

		return tj;
	}

	virtual tdat_loader_worker* clone() const {
		return new tdat_loader_worker();
	}

};



struct _fake_leveldb_imagedb_iterator_inner {
	std::string root_dir;
	const fake_leveldb_imagedb::labeled_data_vec_t* lines;
	fake_leveldb_imagedb::labeled_data_vec_t::const_iterator iter;
	size_t max_open_files;
    size_t tmp_max_open_files;


	std::map<size_t,image_loader_scheduler::key_t> index_to_key;

	struct physical_file_info {
		tdat_loader_scheduler::key_t key;
		fake_leveldb_imagedb::labeled_data_vec_t::const_iterator start_iter;
	};
	std::list< std::string > physical_file_list;
	std::map< std::string, physical_file_info >  physical_file_set;
	std::string current_filename;
	fake_leveldb_imagedb::labeled_data_vec_t::const_iterator next_prefetch_iter;

	size2d canonical_size;
	int color_type;
	
	boost::shared_ptr< image_loader_scheduler > image_s;
	boost::shared_ptr< tdat_loader_scheduler > tdat_s;

	_fake_leveldb_imagedb_iterator_inner( const std::string& root_dir,
			const fake_leveldb_imagedb::labeled_data_vec_t& lines,
			size_t max_open_files, size2d canonical_size, int color_type ) :
				root_dir(root_dir), lines(&lines),
				iter(lines.begin()), max_open_files(max_open_files), tmp_max_open_files(max_open_files),
				next_prefetch_iter(iter), canonical_size(canonical_size), color_type(color_type),
				image_s( new image_loader_scheduler( image_loader_worker(canonical_size, color_type) ) ),
				tdat_s( new tdat_loader_scheduler( tdat_loader_worker(),1 ) )
				 {	}

	_fake_leveldb_imagedb_iterator_inner(
			const _fake_leveldb_imagedb_iterator_inner& _r ) :
				root_dir(_r.root_dir), lines(_r.lines),
				iter(_r.iter), max_open_files(_r.max_open_files), tmp_max_open_files(_r.max_open_files),
				next_prefetch_iter(iter), canonical_size(_r.canonical_size), color_type(_r.color_type),
				image_s( new image_loader_scheduler( image_loader_worker(canonical_size, color_type) ) ),
				tdat_s( new tdat_loader_scheduler( tdat_loader_worker(),1 ) ) {	}

	_fake_leveldb_imagedb_iterator_inner& operator = (
			const _fake_leveldb_imagedb_iterator_inner& _r ) {

		root_dir = _r.root_dir;
		lines    = _r.lines;
		iter     = _r.iter;
		max_open_files = _r.max_open_files;
        tmp_max_open_files = _r.tmp_max_open_files;
		canonical_size = _r.canonical_size;
		color_type = _r.color_type;
		
		image_s.reset( new image_loader_scheduler( image_loader_worker(canonical_size, color_type) ) );
		index_to_key.clear();

		tdat_s.reset( new tdat_loader_scheduler( tdat_loader_worker(),1 ) );

		physical_file_list.clear();
		physical_file_set.clear();

		next_prefetch_iter = iter;

		return *this;
	}

	~_fake_leveldb_imagedb_iterator_inner() {
		tdat_s->stop_processing();
		image_s->stop_processing();
		tdat_s->wait_for_stopped();
		image_s->wait_for_stopped();
	}

	std::pair<cv::Mat,int> get_labeled_image() {

		if (next_prefetch_iter<iter) next_prefetch_iter = iter;

		// ============= prefetch data
		if ( current_filename != iter->first.filename ) {
			current_filename = iter->first.filename;
			// clean at most one file
			while ( physical_file_list.size()>=tmp_max_open_files ) {

				std::map<std::string,_fake_leveldb_imagedb_iterator_inner::physical_file_info>::iterator q =
						physical_file_set.find( physical_file_list.back() );

				boost::shared_ptr<tdat_jobs> tj;
				tj = tdat_s->read( q->second.key );
				for (size_t i=0; i<tj->keys.size(); ++i ) {
					if ( tj->is_first_occurrence[i] )
						image_s->disclaim( tj->keys[i] );
				}

				tdat_s->disclaim( q->second.key );
				physical_file_list.pop_back();
				physical_file_set.erase(q);
			}
            tmp_max_open_files = max_open_files;
			fake_leveldb_imagedb::labeled_data_vec_t::const_iterator p = next_prefetch_iter;
			while( p<lines->end() && physical_file_list.size()<max_open_files ){
				std::string filename = p->first.filename;
				std::map<std::string,_fake_leveldb_imagedb_iterator_inner::physical_file_info>::iterator filename_iter;
				filename_iter = physical_file_set.find( filename );
				if ( filename_iter == physical_file_set.end() ) {
					// new file
					_fake_leveldb_imagedb_iterator_inner::physical_file_info info;
					info.start_iter = p;

					boost::shared_ptr<tdat_info> ti( new tdat_info );
					ti->filename = root_dir + filename;
					ti->image_s  = image_s.get();

					while( p<lines->end() && p->first.filename == filename ) {
						ti->subfile_ranges.push_back( p->first );
						++p;
					}

					info.key = tdat_s->claim( ti );

					physical_file_list.push_front( filename );
					physical_file_set.insert( std::make_pair(filename,info) );

					filename_iter = physical_file_set.find( filename );
				} else {
                    tmp_max_open_files = physical_file_list.size();
					break;
				}
			}
			next_prefetch_iter = p;
		}

		// ====================== read data
		std::string filename = iter->first.filename;
		std::map<std::string,_fake_leveldb_imagedb_iterator_inner::physical_file_info>::iterator filename_iter;
		filename_iter = physical_file_set.find( filename );

		boost::shared_ptr<tdat_jobs> tj;
		tj = tdat_s->read( filename_iter->second.key );
		size_t rloc = iter - filename_iter->second.start_iter;
		cv::Mat I = image_s->read(tj->keys[rloc]);

		return std::make_pair(I,iter->second);
	}


};

// --------------------------------------------------------------

_fake_leveldb_imagedb_iterator::_fake_leveldb_imagedb_iterator( const std::string& root_dir,
		const fake_leveldb_imagedb::labeled_data_vec_t& lines,
		size_t max_open_files, size2d canonical_size, int color_type ) : _h( new
				_fake_leveldb_imagedb_iterator_inner
				( root_dir, lines, max_open_files, canonical_size, color_type ) ) {
}

_fake_leveldb_imagedb_iterator::_fake_leveldb_imagedb_iterator(
		const _fake_leveldb_imagedb_iterator& _r ) : _h( new
				_fake_leveldb_imagedb_iterator_inner( *(_r._h) ) ) {
	;
}

_fake_leveldb_imagedb_iterator& _fake_leveldb_imagedb_iterator::operator = (
		const _fake_leveldb_imagedb_iterator& _r ) {
	*_h = *(_r._h);
	return (*this);
}

_fake_leveldb_imagedb_iterator::~_fake_leveldb_imagedb_iterator() {
	delete _h;
}

bool compare_imagedb_labeled_data_t ( const _fake_leveldb_imagedb_iterator::labeled_data_t& a,
		const _fake_leveldb_imagedb_iterator::labeled_data_t& b ) {
	return a.first.filename < b.first.filename;
}

void _fake_leveldb_imagedb_iterator::SeekToFirst( bool need_shuffle ) {
	_fake_leveldb_imagedb_iterator_inner* old_h = _h;
	if (need_shuffle) {
		auto lines = const_cast<fake_leveldb_imagedb::labeled_data_vec_t*>(_h->lines);
		std::shuffle( lines->begin(), lines->end(), this->a_->rand_g );
		std::stable_sort( lines->begin(), lines->end(), compare_imagedb_labeled_data_t );
	}
	_h = new _fake_leveldb_imagedb_iterator_inner
			( _h->root_dir, *(_h->lines), _h->max_open_files, 
              _h->canonical_size, _h->color_type );
	delete old_h;
}
void _fake_leveldb_imagedb_iterator::Next()  { ++(_h->iter); }
bool _fake_leveldb_imagedb_iterator::Valid() {
	return ( _h->iter>=_h->lines->begin() && (_h->iter)<(_h->lines->end()) );
}

size_t _fake_leveldb_imagedb_iterator::Total() {
	return _h->lines->size();
}

std::pair<cv::Mat,int> _fake_leveldb_imagedb_iterator::get_labeled_image() {
	return _h->get_labeled_image();
}

// --------------------------------------------- fake_leveldb_bbox

fake_leveldb_bbox::Iterator* fake_leveldb_bbox::NewIterator(
		size_t max_open_files, size2d canonical_size, int color_type ) {
    return new Iterator( *root_dir_, *lines_, max_open_files, canonical_size, color_type );
}

int fake_leveldb_bbox::OpenReadOnly( const string& database_path ) {

	string root_dir;
	list< labeled_data_t > lines;

	{	// txt + image database,

		string list_path;
		{
			ifstream db_in( database_path.c_str() );
			if ( db_in.fail() )
				return 1;
			if ( !( getline( db_in, root_dir ) && getline( db_in, list_path ) ) )
				return 2;
		}
		if ( !root_dir.empty() && root_dir[root_dir.length()-1]!='/' )
			root_dir += '/';
		if ( list_path.empty() )
			return 3;
		if ( list_path[0] != '/' ) {
			size_t p = database_path.rfind( '/' );
			if ( p != string::npos )
				list_path = string( database_path, 0, p+1 ) + list_path;
		}

		{
			ifstream infile( list_path.c_str() );
			if (infile.fail())
				return 11;
			string filename;
			int x1, y1, x2, y2, label;
			while (infile >> filename >> y1 >> x1 >> y2 >> x2 >> label) {
				lines.push_back(std::make_pair(
						single_bboxinfo( filename, x1, y1, x2, y2 ), label));
			}
		}
	}

	root_dir_.reset( new string(root_dir) );
	lines_.reset( new labeled_data_vec_t(lines.begin(), lines.end()) );


	return 0;
}


struct bbox_with_img {
	cv::Mat img;
	fake_leveldb_bbox::single_bbox_range box;
};

typedef zeta::restaurant_scheduler< bbox_with_img, cv::Mat > bbox_cropper_scheduler;

class bbox_cropper_worker : public bbox_cropper_scheduler::worker {
	size2d canonical_size;
public:
	bbox_cropper_worker() : canonical_size(0) {}
	bbox_cropper_worker( size2d canonical_size ) : canonical_size(canonical_size) {}
	
	virtual cv::Mat operator () ( bbox_with_img c ) {
		// Be careful !!! this is 0-base
		cv::Mat I = c.img( cv::Rect(c.box.x1,c.box.y1,
				c.box.x2-c.box.x1+1,c.box.y2-c.box.y1+1) );
		cv::Mat J;
		im_resize( I, J, canonical_size );
		return J;
	}
	virtual bbox_cropper_worker* clone() const {
		return new bbox_cropper_worker(canonical_size);
	}
};

struct bboxes_info {
	std::string filename;
	std::vector<fake_leveldb_bbox::single_bboxinfo> boxes;
	bbox_cropper_scheduler* bbox_s;
};

struct bboxes_jobs {
	cv::Mat img;
	vector<bbox_cropper_scheduler::key_t> keys;
};

typedef zeta::restaurant_scheduler< boost::shared_ptr<bboxes_info>,
		boost::shared_ptr<bboxes_jobs> > image_withbbox_loader_scheduler;

class image_withbbox_loader_worker : public image_withbbox_loader_scheduler::worker {
	int color_type;
public:
	image_withbbox_loader_worker() : color_type( fake_leveldb::BGR_COLOR ) {}
	explicit image_withbbox_loader_worker( int color_type ) : color_type( color_type ) {}
	virtual boost::shared_ptr<bboxes_jobs> operator () (
			boost::shared_ptr<bboxes_info> ti ) {

		boost::shared_ptr<bboxes_jobs> tj( new bboxes_jobs );

		tj->img = cv::imread( ti->filename, __translate_colortype2cv(color_type) );

		std::vector<fake_leveldb_bbox::single_bboxinfo>& r = ti->boxes;

		tj->keys.resize( r.size() );
		for ( size_t i=0; i<r.size(); ++i ) {
			bbox_with_img c;
			c.img = tj->img;
			c.box = r[i];
			tj->keys[i] = ti->bbox_s->claim( c );
		}

		return tj;
	}

	virtual image_withbbox_loader_worker* clone() const {
		return new image_withbbox_loader_worker(color_type);
	}

};


struct _fake_leveldb_bbox_iterator_inner {
	std::string root_dir;
	const fake_leveldb_bbox::labeled_data_vec_t* lines;
	fake_leveldb_bbox::labeled_data_vec_t::const_iterator iter;
	size_t max_open_files;
    size_t tmp_max_open_files;


	std::map<size_t,bbox_cropper_scheduler::key_t> index_to_key;

	struct physical_file_info {
		image_withbbox_loader_scheduler::key_t key;
		fake_leveldb_bbox::labeled_data_vec_t::const_iterator start_iter;
	};
	std::list< std::string > physical_file_list;
	std::map< std::string, physical_file_info >  physical_file_set;
	std::string current_filename;
	fake_leveldb_bbox::labeled_data_vec_t::const_iterator next_prefetch_iter;

    size2d canonical_size;
    int color_type;
	
	boost::shared_ptr< bbox_cropper_scheduler > bbox_s;
	boost::shared_ptr< image_withbbox_loader_scheduler > iw_s;

	_fake_leveldb_bbox_iterator_inner( const std::string& root_dir,
			const fake_leveldb_bbox::labeled_data_vec_t& lines,
			size_t max_open_files, size2d canonical_size, int color_type ) :
				root_dir(root_dir), lines(&lines),
				iter(lines.begin()), max_open_files(max_open_files), tmp_max_open_files(max_open_files),
				next_prefetch_iter(iter), canonical_size(canonical_size), color_type(color_type),
				bbox_s( new bbox_cropper_scheduler( bbox_cropper_worker(canonical_size), 1 ) ),
				iw_s( new image_withbbox_loader_scheduler( image_withbbox_loader_worker(color_type), 1 ) )
				{ }

	_fake_leveldb_bbox_iterator_inner(
			const _fake_leveldb_bbox_iterator_inner& _r ) :
				root_dir(_r.root_dir), lines(_r.lines),
				iter(_r.iter), max_open_files(_r.max_open_files), tmp_max_open_files(_r.tmp_max_open_files),
				next_prefetch_iter(iter), canonical_size(_r.canonical_size), color_type(_r.color_type),
				bbox_s( new bbox_cropper_scheduler( bbox_cropper_worker(canonical_size), 1 ) ),
				iw_s( new image_withbbox_loader_scheduler( image_withbbox_loader_worker(color_type), 1 ) )
				 {	}

	_fake_leveldb_bbox_iterator_inner& operator = (
			const _fake_leveldb_bbox_iterator_inner& _r ) {

		root_dir = _r.root_dir;
		lines    = _r.lines;
		iter     = _r.iter;
		max_open_files = _r.max_open_files;
        tmp_max_open_files = _r.tmp_max_open_files;
		
		canonical_size = _r.canonical_size;
		color_type = _r.color_type;

		bbox_s.reset( new bbox_cropper_scheduler( bbox_cropper_worker( canonical_size ), 1 ) );
		index_to_key.clear();

		iw_s.reset( new image_withbbox_loader_scheduler( image_withbbox_loader_worker(color_type), 1 ) );

		physical_file_list.clear();
		physical_file_set.clear();

		next_prefetch_iter = iter;

		return *this;
	}

	~_fake_leveldb_bbox_iterator_inner() {
		iw_s->stop_processing();
		bbox_s->stop_processing();
		iw_s->wait_for_stopped();
		bbox_s->wait_for_stopped();
	}

	std::pair<cv::Mat,int> get_labeled_image() {

		if (next_prefetch_iter<iter) next_prefetch_iter = iter;

		// ============= prefetch data
		if ( current_filename != iter->first.filename ) {
			current_filename = iter->first.filename;
			// clean at most one file
			while ( physical_file_list.size()>=tmp_max_open_files ) {

				std::map<std::string,_fake_leveldb_bbox_iterator_inner::physical_file_info>::iterator q =
						physical_file_set.find( physical_file_list.back() );

				boost::shared_ptr<bboxes_jobs> tj;
				tj = iw_s->read( q->second.key );
				for (size_t i=0; i<tj->keys.size(); ++i ) {
					bbox_s->disclaim( tj->keys[i] );
				}

				iw_s->disclaim( q->second.key );
				physical_file_list.pop_back();
				physical_file_set.erase(q);
			}
            tmp_max_open_files = max_open_files;
			fake_leveldb_bbox::labeled_data_vec_t::const_iterator p = next_prefetch_iter;
			while( p<lines->end() && physical_file_list.size()<max_open_files ){
				std::string filename = p->first.filename;
				std::map<std::string,_fake_leveldb_bbox_iterator_inner::physical_file_info>::iterator filename_iter;
				filename_iter = physical_file_set.find( filename );
				if ( filename_iter == physical_file_set.end() ) {
					// new file
					_fake_leveldb_bbox_iterator_inner::physical_file_info info;
					info.start_iter = p;

					boost::shared_ptr<bboxes_info> ti( new bboxes_info );
					ti->filename = root_dir + filename;
					ti->bbox_s   = bbox_s.get();

					while( p<lines->end() && p->first.filename == filename ) {
						ti->boxes.push_back( p->first );
						++p;
					}

					info.key = iw_s->claim( ti );

					physical_file_list.push_front( filename );
					physical_file_set.insert( std::make_pair(filename,info) );

					filename_iter = physical_file_set.find( filename );
				} else {
                    tmp_max_open_files = physical_file_list.size();
					break;
				}
			}
			next_prefetch_iter = p;
		}

		// ====================== read data
		std::string filename = iter->first.filename;
		std::map<std::string,_fake_leveldb_bbox_iterator_inner::physical_file_info>::iterator filename_iter;
		filename_iter = physical_file_set.find( filename );
        
		boost::shared_ptr<bboxes_jobs> tj;
		tj = iw_s->read( filename_iter->second.key );
		size_t rloc = iter - filename_iter->second.start_iter;
		cv::Mat I = bbox_s->read(tj->keys[rloc]);

		return std::make_pair(I,iter->second);
	}


};

// --------------------------------------------------------------

_fake_leveldb_bbox_iterator::_fake_leveldb_bbox_iterator( const std::string& root_dir,
		const fake_leveldb_bbox::labeled_data_vec_t& lines,
		size_t max_open_files, size2d canonical_size, int color_type ) : _h( new
				_fake_leveldb_bbox_iterator_inner
				( root_dir, lines, max_open_files, canonical_size, color_type ) ) {
}

_fake_leveldb_bbox_iterator::_fake_leveldb_bbox_iterator(
		const _fake_leveldb_bbox_iterator& _r ) : _h( new
				_fake_leveldb_bbox_iterator_inner( *(_r._h) ) ) {
	;
}

_fake_leveldb_bbox_iterator& _fake_leveldb_bbox_iterator::operator = (
		const _fake_leveldb_bbox_iterator& _r ) {
	*_h = *(_r._h);
	return (*this);
}

_fake_leveldb_bbox_iterator::~_fake_leveldb_bbox_iterator() {
	delete _h;
}

bool compare_bbox_data_t ( const _fake_leveldb_bbox_iterator::labeled_data_t& a,
		const _fake_leveldb_bbox_iterator::labeled_data_t& b ) {
	return a.first.filename < b.first.filename;
}


void _fake_leveldb_bbox_iterator::SeekToFirst( bool need_suffle ) {

	if ( need_suffle ) {
		auto lines = const_cast<_fake_leveldb_bbox_iterator::labeled_data_vec_t*>(_h->lines);
		std::shuffle( lines->begin(), lines->end(), this->a_->rand_g );
		std::stable_sort( lines->begin(), lines->end(), compare_bbox_data_t );
	}

	_fake_leveldb_bbox_iterator_inner* old_h = _h;
	_h = new _fake_leveldb_bbox_iterator_inner
			( _h->root_dir, *(_h->lines), _h->max_open_files, 
              _h->canonical_size, _h->color_type );
	delete old_h;
}
void _fake_leveldb_bbox_iterator::Next()  { ++(_h->iter); }
bool _fake_leveldb_bbox_iterator::Valid() {
	return ( _h->iter>=_h->lines->begin() && (_h->iter)<(_h->lines->end()) );
}

size_t _fake_leveldb_bbox_iterator::Total() {
	return _h->lines->size();
}

std::pair<cv::Mat,int> _fake_leveldb_bbox_iterator::get_labeled_image() {
	return _h->get_labeled_image();
}



}

#endif
