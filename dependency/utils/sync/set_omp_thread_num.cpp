#ifndef __APPLE__
#include <omp.h>
#endif
#include<mex.h>

void mexFunction( int nlhs, mxArray *plhs[], 
        int nrhs, const mxArray *prhs[]) {
#ifndef __APPLE__

    size_t num_threads = 0;
    //omp_set_dynamic(0); 
    if (nrhs>1)
        mexErrMsgTxt("At most one arguments");
    else if (nrhs==1)
        num_threads = mxGetScalar( prhs[0] );
    else {
        num_threads = omp_get_num_procs();
    }
    if ( num_threads>0 )
        omp_set_num_threads( num_threads );

#endif

}

