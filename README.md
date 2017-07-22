# Introduction

This repository implements DBNet, proposed in the following paper:

> Yuting Zhang, Luyao Yuan, Yijie Guo, Zhiyuan He, I-An Huang, Honglak Lee, “Discriminative Bimodal Networks for Natural Language Visual Localization and Detection”, *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2017. **Spotlight**Remark:

* This code is the initial implementation of DBNet.
* A Python+TensorFlow implementation is available at https://github.com/yuanluya/dbnet_tensorflow
* An independent development and evaluation toolbox for the data used in the DBNet paper is available at https://github.com/YutingZhang/nlvd_evaluation . However, it is not used in this repository, because it is developed in a later time.

# Compilation

This code is based on Caffe and MATLAB (including C++ code).

## Compile Caffe

The custom version of Caffe is at `./caffe`. Please compile it by yourself (refer to the official guide http://caffe.berkeleyvision.org/installation.html ). C++11 compatible compiler is needed. You will need to successfully run the following commands:

	make
	make matcaffe

## Compile MEX (C++ in MATLAB)

Start MATLAB in the root folder of the code. 
Run the following command:

	compile_mex

# Setting up data 

Please download and set up all data as follows.

## Visual Genome annotations

The following link contains Visual Genome annotations in MATLAB format. Please download and extract it *somewhere* (you can specify the path later in the config file). 

* http://www.ytzhang.net/files/dbnet/data-caffe/edgebox.tar.gz

The annotations have been cleaned up as described in the paper. Extra annotations are provided for text similarity.

Please download the Visual Genome images from the official site. 

* https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
* https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip

After extracting the annotations and images, please do the following:

1. Start MATLAB at the root folder of the code
2. Run `global_settings;`
3. Exit MATLAB
4. Update the Visual Genome paths in `./system/global_settings.m`

## Networks and pre-trained models

The following link contains network definitions, pre-trained models, and test results for VGGNet-16 based DBNet. Please download and extract it *at the root folder of the code*. 

* http://www.ytzhang.net/files/dbnet/data-caffe/vgg16_models.tar.gz

## Precomputed region proposal

The following link contains precomputed region proposal for all Visual Genome images using EdgeBox. Please download and extract it *somewhere* (you can specify the path later in the config file).

* http://www.ytzhang.net/files/dbnet/data-caffe/vg_annotations.tar.gz

# Run experiments

* Start MATLAB at the root folder of the code
* `cd ./exp/vgg16/script`

You will find the experiment have 4 phases, where `param_phase?` (`?=1,2,3,4`) defines the parameters and input/output directories, and `run_phase?('GPU_ID',0)` (`?=1,2,3,4`) runs the experiment for each phase, e.g., on GPU 0.

The 4 phases are as follows:

* Phase 1: Train with Visual Genome region phrase annotations. The linguistic and discriminative pathways are trained from scratch. The visual pathway is fixed as the faster R-CNN model trained on PASCAL VOC (07+12). The model will be saved in `./exp/vgg16/cache/Train_phase1`
* Phase 2: Train with Visual Genome region phrase annotations. The entire network is finetuned with the base learning rate. The model will be saved in `./exp/vgg16/cache/Train_phase2`
* Phase 3: Further finetune the whole network with smaller learning rate (x10 smaller). The model will be saved in `./exp/vgg16/cache/Train_phase3`
* Phase 4: Test the model learned in Phase 3 for visual entity localization using object-relationship-object query text on 5000 test images. The test results will be saved in `./exp/vgg16/cache/Test`

## Run a phase from scratch

Note that the downloaded data include cache for all phases. When you start the experiment, it will resume from the current cache. In particular,

* For training, it will resume from the latest checkpoint.
* For testing, it will say the test results have already existed and do nothing. 

To run a phase from scratch, please remove the files in the corresponding output folders (as previously specified).

## Snapshot and resume training

The training script automatically does snapshotting in every 20,000 iterations. This interval is defined at the beginning of `./pipeline/pipTrain.m`

The training will not terminate automatically (the default maximum number of iterations is set to `Inf`). You can use `Ctrl+C` to pause the training in MATLAB. Then,

* You can run `npl_snapshot()` to snapshot the current state of the model.
* You can run `npl_resume()` to resume the training. 

## Interrupt and resume testing

Testing can also be interrupted (by `Ctrl+C`) and resumed (by running the test phase again) at any time.

Note that an index file `./exp/vgg16/cache/Test/_index.lock` is used to track the testing progress. Any image that has been tested will not be tested again. You can remove the index file to let the script try to test all images again. 

## Override default parameters

To run a phase using `run_phase?(___)`, you can override any default parameters (without changing the source code) by providing the parameters in the command line, such as

	run_phase1('GPU_ID',0,'Train_SnapshotFrequency',5000,'Train_MaxIteration',3e5,'BaseModel_BatchSize',1,'ConvFeatNet_BatchSize',1)

## TODOs (will be available soon)

* Detection demo on a few sample images
* Detection benchmarking phase
* Pretrained model for ResNet-101

