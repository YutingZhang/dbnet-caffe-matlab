
// -------------------------- Generic part
#include <boost/filesystem.hpp>
#include <exception>
#include <cstdlib>

using namespace std;
// --------------------------- MEX part

#include<mex.h>
void mexFunction( int nlhs, mxArray *plhs[], 
        int nrhs, const mxArray *prhs[])
{
    if ( nrhs == 0 ) {
        mexPrintf(
                "Usage: abs_path = absolute_path( rel_path, [base_path] )\n"
                );
        return;
    } else if ( nrhs > 2 ) {
        mexErrMsgTxt("One or two input argument");
    }

    if ( nlhs>1 )
        mexErrMsgTxt("Exactly one output");

    if (!mxIsChar(prhs[0]))
        mexErrMsgTxt("All the arguments should be strings.");

    string base_path_str;
    
    char* src_ = mxArrayToString(prhs[0]);
    string src(src_);
    mxFree(src_);
    
    if ( (src.length()==1 && src[0] == '~') || 
            (src.length()>=2 && src[0] == '~' && src[1] == '/') ) {
        const string home_dir = std::getenv("HOME");
        src = home_dir + src.substr(1);
    }
    
    if (nrhs>1) {
        if (!mxIsChar(prhs[1]))
            mexErrMsgTxt("All the arguments should be strings.");
        char* base_ = mxArrayToString(prhs[1]);
        base_path_str = base_;
        mxFree(base_);
    }
    
    string abs_path_str;
    try {
        boost::filesystem::path abs_path;
        boost::filesystem::path rel_path(src);
        if (base_path_str.empty())
            abs_path = boost::filesystem::absolute( src );
        else
            abs_path = boost::filesystem::absolute( src, boost::filesystem::path(base_path_str) );
        abs_path_str = abs_path.native();
    } catch ( const exception& e ) {
        mexErrMsgTxt( e.what() );
    }

    plhs[0] = mxCreateString( abs_path_str.c_str() );


}

