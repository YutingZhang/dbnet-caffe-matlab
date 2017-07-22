
// -------------------------- Generic part
#include <fstream>
#include <boost/interprocess/sync/file_lock.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <exception>
#include <vector>
#include <string>
#include <limits>
#include <ctime>
#include <iostream>

using namespace std;
using namespace boost::interprocess;

enum FLAGS {
    MODE_INC         = 0,
    MODE_RESET,
    MODE_STATUS,
    MODE_ALIVE,
    MODE_DONE,
};

enum {
    STAGE_INCREMENTAL = 1,
    STAGE_CLEANUP     = 2
};

struct file_header_t {
    
    long unsigned int total;         // total tasks
    long unsigned int next_task;     // the next pointer position
    long unsigned int complete;      // the number of tasks that are already complete
    long unsigned int left_total;    // the total number of tasks left for stage2
    long unsigned int next_start_time;
    long unsigned int last_task;     // 
    
    int stage() {
        if ( left_total || !total ) return STAGE_CLEANUP;
        else return STAGE_INCREMENTAL;
    }
    
};

struct common_element_t {
    long unsigned int start_time;
    
    static long unsigned int task_pending_value() { return 0; }
    static long unsigned int task_complete_value() { 
        return std::numeric_limits<long unsigned int>::max(); 
    }
    
    common_element_t() : start_time(0) {}
};

#define TASK_PENDING  (common_element_t::task_pending_value() )
#define TASK_COMPLETE (common_element_t::task_complete_value())


struct stage1_element_t : public common_element_t { };

struct stage2_element_t : public common_element_t  {
    long unsigned int task_index;
    stage2_element_t() : task_index(0) {}
    stage2_element_t( const stage2_element_t& _r ) : 
        common_element_t(_r), task_index(_r.task_index) {}
    stage2_element_t( const common_element_t& _r ) : 
        common_element_t(_r) {}
};

template<class T>
void read_from_stream( std::istream& in, T* dst, size_t count = 1 ) {
    const size_t s = sizeof(T) * count;
    char* d = reinterpret_cast<char*>(dst);
    in.read( d, s );
}

template<class T>
void write_to_stream( std::ostream& out, const T* src, size_t count = 1 ) {
    const size_t s = sizeof(T) * count;
    const char* d = reinterpret_cast<const char*>(src);
    out.write( d, s );
}

long unsigned int atomic_index( file_header_t* dst_header ,
        const char* index_file_path, int flags, 
        long unsigned int task_total, double task_time_out, 
        long unsigned int task_id ) {
    {
        try {
            file_lock f_lock(index_file_path);
            scoped_lock<file_lock> e_lock(f_lock);
        } catch (...) {
            ofstream out( index_file_path, ofstream::out | 
                    ofstream::app | ofstream::binary );
            if (!out)
                throw "Cannot write to the index file";
        }
    }

    file_header_t header_;
    file_header_t* header_ptr = (dst_header)?(dst_header):(&header_);
    file_header_t& header = *header_ptr;
            
    long unsigned int cur_task = 0lu;
    std::vector<stage1_element_t> s1vec;
    std::vector<stage2_element_t> s2vec;
    bool force_init = false;
    file_lock f_lock(index_file_path);
    {
        scoped_lock<file_lock> e_lock(f_lock);
        fstream cf( index_file_path, ofstream::out | 
                ofstream::in | ofstream::binary  );
        
        switch ( flags ) {
            case MODE_RESET:
                force_init = true;
            case MODE_STATUS:
            case MODE_INC:
            case MODE_ALIVE:
            case MODE_DONE:
                read_from_stream( cf, &header );
                if ( force_init || cf.fail() ) {
                    //// need to create a new file
                    header.total = task_total;
                    header.complete   = 0lu;
                    header.left_total = 0lu;
                    if (task_total) {
                        header.next_task  = 1lu;
                        header.next_start_time = TASK_PENDING;
                        header.last_task  = TASK_COMPLETE;
                    } else {
                        header.next_task  = 0lu;
                        header.next_start_time = TASK_COMPLETE;
                        header.last_task  = 0lu;
                    }
                    
                    s1vec.resize( task_total );
                    
                    cf.close();
                    cf.open( index_file_path, fstream::out | ofstream::binary );
                    write_to_stream( cf, &header );
                    write_to_stream( cf, &(s1vec[0]), task_total );
                }
                
                if (task_total==0)
                    task_total = header.total;
                else if (task_total != header.total)
                    throw "Total number of tasks are inconsistent";
                
                if ( flags == MODE_STATUS ) {
                    break;
                } else if ( flags == MODE_INC || flags == MODE_RESET ) {
                    if (header.stage() == STAGE_INCREMENTAL) {
                        cur_task = (header.next_task++);
                        header.last_task = cur_task;
                        
                        {   //update start time
                            cf.seekp(sizeof(header)+sizeof(stage1_element_t)*(cur_task-1), 
                                    ios_base::beg);
                            stage1_element_t ts;
                            ts.start_time = time(NULL);
                            write_to_stream( cf, &ts );
                        }


                        // start a new task, and update task start time
                        if ( header.next_task<=header.total ) {
                            header.next_start_time = TASK_PENDING;
                            
                            cf.seekp(0, ios_base::beg);
                            write_to_stream( cf, &header );

                        } else {
                            // enter stage_cleanup
                            
                            s1vec.resize( task_total );
                            cf.seekg( sizeof(header), ios_base::beg );
                            read_from_stream( cf, &(s1vec[0]), task_total );
                            s2vec.clear();
                            header.next_start_time = TASK_COMPLETE;
                            header.next_task        = 0;

                            for ( size_t i=0; i<s1vec.size(); ++i ) {
                                if ( s1vec[i].start_time != TASK_COMPLETE ) {
                                    stage2_element_t et(s1vec[i]);
                                    et.task_index = i+1;
                                    s2vec.push_back( et );
                                    
                                    if (et.start_time < header.next_start_time) {
                                        header.next_start_time = et.start_time;
                                        header.next_task = i+1;
                                    }
                                }
                            }
                            header.left_total = s2vec.size();
                            
                            cf.seekp(0, ios_base::beg);
                            write_to_stream( cf, &header );
                            write_to_stream( cf, &(s2vec[0]), s2vec.size() );
                        }
                    } else if (header.stage() == STAGE_CLEANUP) {
                        time_t now = time(NULL);
                        double elapsed_time = difftime( now, header.next_start_time );
                        
                        if ( !header.next_task ) break; // no additinal
                        if ( elapsed_time<task_time_out ) break; // not ready, cannot decide
                        
                        cur_task = header.next_task;
                        header.last_task = cur_task;
                        
                        s2vec.resize( header.left_total );
                        cf.seekg(sizeof(header),ios_base::beg);
                        read_from_stream( cf, &(s2vec[0]), s2vec.size() );
                        
                        header.next_start_time = TASK_COMPLETE;
                        header.next_task       = 0;
                        for ( size_t i=0; i<s2vec.size(); ++i ) {
                            if (s2vec[i].task_index == cur_task) {
                                //update start time
                                s2vec[i].start_time = now;
                                
                                cf.seekp(sizeof(header)+sizeof(stage2_element_t)*i, 
                                        ios_base::beg);
                                write_to_stream( cf, &(s2vec[i]) );                                
                            }
                            if (s2vec[i].start_time < header.next_start_time) {
                                header.next_start_time = s2vec[i].start_time;
                                header.next_task = s2vec[i].task_index;
                            }
                        }
                        
                        cf.seekp(0,ios_base::beg);
                        write_to_stream( cf, &header );
                        
                    } else {
                        throw "Internal error: Unrecognized stage for INC";
                    }
                } else {    // MODE_ALIVE, MODE_DONE
                    
                    if ( task_id<1 || task_id>header.total )
                        throw "task_id is out of range";
                    
                    long unsigned int nt;
                    if ( flags == MODE_ALIVE ) {
                        nt = time(NULL);
                    } else if (flags == MODE_DONE) {
                        nt = TASK_COMPLETE;
                    } else
                        throw "Internal error: Unknown flags";
                    
                    bool is_success = false;
                    
                    if (header.stage() == STAGE_INCREMENTAL) {
                        stage1_element_t et;
                        cf.seekg(sizeof(header)+sizeof(stage1_element_t)*(task_id-1), 
                                ios_base::beg);
                        read_from_stream( cf, &et );
                        
                        if (nt>et.start_time) {
                            et.start_time = nt;
                            cf.seekp(sizeof(header)+sizeof(stage1_element_t)*(task_id-1), 
                                    ios_base::beg);
                            write_to_stream( cf, &et );
                            is_success = true;
                        }
                        
                    } else if (header.stage() == STAGE_CLEANUP) {
                        
                        s2vec.resize( header.left_total );
                        cf.seekg(sizeof(header),ios_base::beg);
                        read_from_stream( cf, &(s2vec[0]), s2vec.size() );
                        
                        for ( size_t i=0; i<s2vec.size(); ++i ) {
                            if (s2vec[i].task_index == task_id) {
                                stage2_element_t& et = s2vec[i];
                                if (nt>et.start_time) {
                                    et.start_time = nt;
                                    cf.seekp(sizeof(header)+sizeof(stage2_element_t)*i, 
                                            ios_base::beg);
                                    write_to_stream( cf, &et );
                                    is_success = true;
                                }
                            }
                        }
                        
                        if (is_success && header.next_task == task_id ) {
                            header.next_start_time = TASK_COMPLETE;
                            header.next_task       = 0;
                            for ( size_t i=0; i<s2vec.size(); ++i ) {
                                if (s2vec[i].start_time < header.next_start_time) {
                                    header.next_start_time = s2vec[i].start_time;
                                    header.next_task = s2vec[i].task_index;
                                }
                            }
                            cf.seekp(0,ios_base::beg);
                            write_to_stream( cf, &header );
                        }
                        
                    } else {
                        throw "Internal error: Unrecognized stage for ACTIVAE and COMPLETE";
                    }
                    
                    if (is_success) {
                        cur_task = task_id; // flag for success
                        if (flags == MODE_DONE ) {
                            ++header.complete;
                            cf.seekp(0, ios_base::beg);
                            write_to_stream( cf, &header );
                        }
                    }
                }
                break;
            default:
                throw "Unknown flags";
        }
    }
    return cur_task;
}

// --------------------------- MEX part

#ifndef BUILD_COMMAND_LINE

#include<mex.h>
void mexFunction( int nlhs, mxArray *plhs[], 
        int nrhs, const mxArray *prhs[])
{
    if ( nrhs == 0 ) {
        mexPrintf(
                "Usage: [cur_task, total,next_task,complete,left_total,next_start_time,last_task] = \n"
                "           atomic_index_( index_file_path, cmd, task_total, task_time_out, task_id )\n"
                "  cmd: 'inc'/'next' (default), 'reset', 'status', 'alive', 'done'\n"
                );
        return;
    } else if ( nrhs != 5 ) {
        mexErrMsgTxt("Exact 5 input arguments");
    }

    if (!mxIsChar(prhs[0]))
        mexErrMsgTxt("The first argument should be a string.");

    int flags;
    if (!mxIsChar(prhs[1]))
        mexErrMsgTxt("The second argument should be a string.");
    char* cmd_ = mxArrayToString(prhs[1]);
    string cmd(cmd_);
    mxFree(cmd_);
    if (cmd == "inc" || cmd == "next")
        flags = MODE_INC;
    else if (cmd == "reset")
        flags = MODE_RESET;
    else if (cmd == "status")
        flags = MODE_STATUS;
    else if (cmd == "alive")
        flags = MODE_ALIVE;
    else if (cmd == "done")
        flags = MODE_DONE;
    else
        mexErrMsgTxt("Unrecognized cmd.");


    
    char* fn_ = mxArrayToString(prhs[0]);
    string fn(fn_);
    mxFree(fn_);

    long unsigned int task_total    = static_cast<long unsigned int>( mxGetScalar(prhs[2]) );
    long unsigned int task_time_out = mxGetScalar(prhs[3]);
    long unsigned int task_id       = static_cast<long unsigned int>( mxGetScalar(prhs[4]) );
    
    file_header_t dst_header;
    unsigned long int cur_task;
    try {
        cur_task = atomic_index( &dst_header, fn.c_str(), flags, 
                task_total, task_time_out, task_id );
    } catch( const std::exception& e ) {
        mexErrMsgTxt( e.what() );
    } catch( const char* e_str ) {
        mexErrMsgTxt( e_str );
    } catch( const string& e_str ) {
        mexErrMsgTxt( e_str.c_str() );
    }
    
    plhs[0] = mxCreateDoubleScalar( static_cast<double>(cur_task) );
    plhs[1] = mxCreateDoubleScalar( static_cast<double>(dst_header.total) );
    plhs[2] = mxCreateDoubleScalar( static_cast<double>(dst_header.next_task) );
    plhs[3] = mxCreateDoubleScalar( static_cast<double>(dst_header.complete) );
    plhs[4] = mxCreateDoubleScalar( static_cast<double>(dst_header.left_total) );
    plhs[5] = mxCreateDoubleScalar( static_cast<double>(dst_header.next_start_time) );
    plhs[6] = mxCreateDoubleScalar( static_cast<double>(dst_header.last_task) );

}

#else

#include <boost/lexical_cast.hpp>

int main( int argn, char** argc ) {
    
    using boost::lexical_cast;
    using boost::bad_lexical_cast;
    
    if ( argn <= 1 ) {
        cerr << "USAGE: atomic_index_ index_file_path cmd task_total task_time_out task_id" << endl;
        return 0;
    } else if ( argn != 6 ) {
        cerr << "Exact 5 arguments" << endl;
        return 0;
    }


    int flags;

    string cmd(argc[2]);
    if (cmd == "inc" || cmd == "next")
        flags = MODE_INC;
    else if (cmd == "reset")
        flags = MODE_RESET;
    else if (cmd == "status")
        flags = MODE_STATUS;
    else if (cmd == "alive")
        flags = MODE_ALIVE;
    else if (cmd == "done")
        flags = MODE_DONE;
    else {
        cerr << "Unrecognized cmd." << endl;
        return 0;
    }

    string fn(argc[1]);

    long unsigned int task_total    = lexical_cast<long unsigned int>( argc[3] );
    long unsigned int task_time_out = lexical_cast<long unsigned int>( argc[4] );
    long unsigned int task_id       = lexical_cast<long unsigned int>( argc[5] );
    
    file_header_t dst_header;
    unsigned long int cur_task;
    try {
        cur_task = atomic_index( &dst_header, fn.c_str(), flags, 
                task_total, task_time_out, task_id );
    } catch( const std::exception& e ) {
        cerr << e.what(); return 0;
    } catch( const char* e_str ) {
        cerr << e_str; return 0;
    } catch( const string& e_str ) {
        cerr << e_str; return 0;
    }
    
    cout << cur_task << endl;
    cout << dst_header.total << endl;
    cout << dst_header.next_task << endl;
    cout << dst_header.complete << endl;
    cout << dst_header.left_total << endl;
    cout << dst_header.next_start_time << endl;
    cout << dst_header.last_task << endl;

    return 1;
}

#endif

