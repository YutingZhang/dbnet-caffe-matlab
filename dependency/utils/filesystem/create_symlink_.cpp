
// -------------------------- Generic part
#include <boost/filesystem.hpp>
#include <exception>

using namespace std;
// --------------------------- MEX part

#include<mex.h>
void mexFunction( int nlhs, mxArray *plhs[], 
        int nrhs, const mxArray *prhs[])
{
    if ( nrhs == 0 ) {
        mexPrintf(
                "Usage: error_code = create_symlink( src, dst )\n"
                );
        return;
    } else if ( nrhs != 2 ) {
        mexErrMsgTxt("Exact two input argument");
    }

    if ( nlhs>1 )
        mexErrMsgTxt("Exactly one output");

    if (!mxIsChar(prhs[0]) || !mxIsChar(prhs[1]))
        mexErrMsgTxt("All the arguments should be strings.");

    
    char* src_ = mxArrayToString(prhs[0]);
    string src(src_);
    mxFree(src_);

    char* dst_ = mxArrayToString(prhs[1]);
    string dst(dst_);
    mxFree(dst_);
    
    int e;
    try {
        boost::system::error_code ec;
        boost::filesystem::create_symlink( src, dst, ec );
        e = ec.value();
    } catch ( const exception& e ) {
        mexErrMsgTxt( e.what() );
    }

    double ed = static_cast<double>(e);
    plhs[0] = mxCreateDoubleScalar( ed );

}

