
// -------------------------- Generic part
#include <boost/filesystem.hpp>
#include <ctime>
#include <exception>
#include <fstream>

using namespace std;
// --------------------------- MEX part

#include<mex.h>
void mexFunction( int nlhs, mxArray *plhs[], 
        int nrhs, const mxArray *prhs[])
{
    if ( nrhs == 0 ) {
        mexPrintf(
                "Usage: touch_file( path )\n"
                );
        return;
    } else if ( nrhs != 1 ) {
        mexErrMsgTxt("Exact one input argument");
    }

    if ( nlhs>0 )
        mexErrMsgTxt("no output");

    if (!mxIsChar(prhs[0]))
        mexErrMsgTxt("All the arguments should be strings.");

    
    char* src_ = mxArrayToString(prhs[0]);
    string src(src_);
    mxFree(src_);
    
    try {
        boost::system::error_code ec;
        //bool ex = boost::filesystem::exists( src );
        bool ex;
        {
            ifstream in(src.c_str(), std::ios_base::in & 
                    std::ios_base::binary);
            ex = in.is_open();
        }
        if (ex) {
            time_t timer;
            time(&timer);
            boost::filesystem::last_write_time( src, timer );
        } else {
            ofstream out(src.c_str(), std::ios_base::out & 
                    std::ios_base::binary & std::ios_base::app);
        }
    } catch ( const exception& e ) {
        mexErrMsgTxt( e.what() );
    }


}

