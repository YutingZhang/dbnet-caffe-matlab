## Refer to http://caffe.berkeleyvision.org/installation.html
# Contributions simplifying and improving our build system are welcome!

# cuDNN acceleration switch (uncomment to build with cuDNN).
USE_CUDNN := 1

# CPU-only switch (uncomment to build without GPU support).
# CPU_ONLY := 1

# uncomment to disable IO dependencies and corresponding data layers
# USE_OPENCV := 0
# USE_LEVELDB := 0
# USE_LMDB := 0

# uncomment to allow MDB_NOLOCK when reading LMDB files (only if necessary)
#	You should not set this flag if you will be reading LMDBs with any
#	possibility of simultaneous read and write
# ALLOW_LMDB_NOLOCK := 1

# Uncomment if you're using OpenCV 3
# OPENCV_VERSION := 3

# To customize your choice of compiler, uncomment and set the following.
# N.B. the default for Linux is g++ and the default for OSX is clang++
# CUSTOM_CXX := g++

# CUDA directory contains bin/ and lib/ directories that we need.
CUDA_DIR :=$(shell echo $$CUDA_ROOT)
# On Ubuntu 14.04, if cuda tools are installed via
# "sudo apt-get install nvidia-cuda-toolkit" then use this instead:
# CUDA_DIR := /usr

# CUDA architecture setting: going with all of them (up to CUDA 5.5 compatible).
# For the latest architecture, you need to install CUDA >= 6.0 and uncomment
# the *_50 lines below.
CUDA_ARCH := -gencode arch=compute_20,code=sm_20 \
		-gencode arch=compute_20,code=sm_21 \
		-gencode arch=compute_30,code=sm_30 \
		-gencode arch=compute_35,code=sm_35 \
		-gencode arch=compute_50,code=sm_50 \
		-gencode arch=compute_50,code=compute_50

# BLAS choice:
# atlas for ATLAS (default)
# mkl for MKL
# open for OpenBlas
BLAS := mkl
# Custom (MKL/ATLAS/OpenBLAS) include and lib directories.
# Leave commented to accept the defaults for your choice of BLAS
# (which should work)!
# MKL directory contains include/ and lib/ directions that we need.
MKL_DIR := $(shell echo $$MKL_ROOT)
BLAS_INCLUDE ?= $(MKL_DIR)/include
BLAS_LIB ?= $(MKL_DIR)/lib $(MKL_DIR)/lib/intel64

# This is required only if you will compile the matlab interface.
# MATLAB directory should contain the mex binary in /bin.
MATLAB_DIR := $(shell dirname `which matlab`)/..
# MATLAB_DIR := /Applications/MATLAB_R2012b.app

# NOTE: this is required only if you will compile the python interface.
# We need to be able to find Python.h and numpy/arrayobject.h.
PYTHON_DIR := $(shell dirname `which python`)/..
PYTHON_MAIN_VERSION := $(shell which python | sed -e 's/^.*\/\([0-9]*\.[0-9]*\)\.[0-9]*.*$$/\1/')
PYTHON_INCLUDE := $(PYTHON_DIR)/include/python$(PYTHON_MAIN_VERSION) \
	$(PYTHON_DIR)/lib/python$(PYTHON_MAIN_VERSION)/site-packages/numpy/core/inlcude

# Anaconda Python distribution is quite popular. Include path:
# PYTHON_INCLUDE := $(HOME)/anaconda/include \
		# $(HOME)/anaconda/include/python2.7 \
		# $(HOME)/anaconda/lib/python2.7/site-packages/numpy/core/include

# We need to be able to find libpythonX.X.so or .dylib.
PYTHON_LIB := $(PYTHON_DIR)/lib
# PYTHON_LIB := /usr/lib
# PYTHON_LIB := $(HOME)/anaconda/lib

# Whatever else you find you need goes here.
INCLUDE_DIRS := $(PYTHON_INCLUDE)
LIBRARY_DIRS := $(PYTHON_LIB)

BUILD_DIR := build
DISTRIBUTE_DIR := distribute

# Uncomment for debugging. Does not work on OSX due to https://github.com/BVLC/caffe/issues/171
# DEBUG := 1

# The ID of the GPU that 'make runtest' will use to run unit tests.
TEST_GPUID := 0

# enable pretty build (comment to see full commands)
Q ?= @
