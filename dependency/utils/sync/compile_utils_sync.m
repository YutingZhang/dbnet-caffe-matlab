function compile_utils_sync()

CUR_DIR = pwd;

returnToCurDir = onCleanup( @() cd(CUR_DIR) );

THIS_DIR = fileparts(which(mfilename('fullpath')));
cd(THIS_DIR);

mex -O atomic_index_.cpp

if ismac
    mex -O set_omp_thread_num.cpp
else
    mex CXXFLAGS="-fopenmp \$CXXFLAGS" LDFLAGS="-fopenmp \$LDFLAGS" -O set_omp_thread_num.cpp
end
