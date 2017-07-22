PARAM = struct();
PARAM.BaseModel_BatchSize = 2;
PARAM.BaseModel_LR = 0.01;
PARAM.BaseModel_Solver.momentum = 0.9;
PARAM.BaseModel_Solver.type = 'SGD';
PARAM.BaseModel_Solver.weight_decay = 0.0005;
PARAM.BatchStacker_Name = 'vgg-faster-rcnn';
PARAM.ConvFeatNet_BatchSize = 2;
PARAM.ConvFeatNet_LR = 0.0001;
PARAM.ConvFeatNet_Name = 'vgg-faster-rcnn';
PARAM.ConvFeatNet_Param = [];
PARAM.ConvFeatNet_Solver.momentum = 0.9;
PARAM.ConvFeatNet_Solver.type = 'SGD';
PARAM.ConvFeatNet_Solver.weight_decay = 0.0005;
PARAM.ConvFeatNet_Updatable = 1;
PARAM.DataSet_Name = 'vg';
PARAM.DataSet_Tag = 'main';
PARAM.DataSet_VG_SplitType = 'densecap';
PARAM.PairNet_BatchSize = 512;
PARAM.PairNet_LR = 1e-05;
PARAM.PairNet_Name = 'dyfilter-dropout-layer0';
PARAM.PairNet_Param.FeatureDim = 4096;
PARAM.PairNet_Param.ImageDropout = 0.3;
PARAM.PairNet_Param.PositiveWeight = 1;
PARAM.PairNet_Param.TextDropout = 0.3;
PARAM.PairNet_Param.WeightDecay = 1e-08;
PARAM.PairNet_Solver.delta = 1e-08;
PARAM.PairNet_Solver.momentum = 0.9;
PARAM.PairNet_Solver.momentum2 = 0.999;
PARAM.PairNet_Solver.type = 'Adam';
PARAM.PairNet_Solver.weight_decay = 0.0005;
PARAM.PairNet_Updatable = 1;
PARAM.RegionFeatNet_BatchSize = 256;
PARAM.RegionFeatNet_LR = 0.0001;
PARAM.RegionFeatNet_Name = 'vgg-faster-rcnn';
PARAM.RegionFeatNet_Param = [];
PARAM.RegionFeatNet_Solver.momentum = 0.9;
PARAM.RegionFeatNet_Solver.type = 'SGD';
PARAM.RegionFeatNet_Solver.weight_decay = 0.0005;
PARAM.RegionFeatNet_Updatable = 1;
PARAM.RegionProposalNet_BatchSize = 2;
PARAM.RegionProposalNet_Name = 'edgebox';
PARAM.RegionProposalNet_Param.Maximum_BoxNum = 1000;
PARAM.RegionProposalNet_Type = 'cached';
PARAM.RegionProposalNet_Updatable = 0;
PARAM.Sampler_GTNearby_Threshold = 0.1;
PARAM.Sampler_Image_HardNegativeNum = 0;
PARAM.Sampler_LabelRemapping = 'threshold';
PARAM.Sampler_LabelThreshold_Negative = 0.1;
PARAM.Sampler_LabelThreshold_Positive = 0.9;
PARAM.Sampler_NegativeTextFromSameImage = 1;
PARAM.Sampler_ProposalRadomNum = 50;
PARAM.Sampler_ProposalTopNum = 50;
PARAM.Sampler_RandomNegativeTextNum = 30;
PARAM.Sampler_Region_HardNegativeNum = 0;
PARAM.Sampler_TextSimilarityThreshold = 1;
PARAM.TextFeatNet_BatchSize = 128;
PARAM.TextFeatNet_LR = 1e-05;
PARAM.TextFeatNet_Name = 'cybermail-lrelu-dy2';
PARAM.TextFeatNet_Param.DyFilter_Dim = 4096;
PARAM.TextFeatNet_Solver.delta = 1e-08;
PARAM.TextFeatNet_Solver.momentum = 0.9;
PARAM.TextFeatNet_Solver.momentum2 = 0.999;
PARAM.TextFeatNet_Solver.type = 'Adam';
PARAM.TextFeatNet_Solver.weight_decay = 0.0005;
PARAM.TextFeatNet_Updatable = 1;
PARAM.TrainSnapshot_External = 0;
PARAM.TrainSnapshot_Iter = 240000;
PARAM.Train_DataSubset = 'train';
PARAM.Train_Finetune = 1;
PARAM.Train_ImageLimit = Inf;
PARAM.base_PrepDataset = struct([]);
PARAM.base_Train = struct([]);
PARAM.base_TrainSnapshot = struct([]);

PRE_PARAM = cmd2struct('param_phase2');
SPECIFIC_DIRS = struct();
SPECIFIC_DIRS.BatchStacker = 'BatchStacker';
SPECIFIC_DIRS.ConvFeatNet = 'ConvFeatNet';
SPECIFIC_DIRS.RegionFeatNet = 'RegionFeatNet';
SPECIFIC_DIRS.TextFeatNet = 'TextFeatNet';
SPECIFIC_DIRS.PairNet = 'PairNet';
SPECIFIC_DIRS.RegionProposalNet = 'RegionProposalNet';
SPECIFIC_DIRS.InitSnapshot = 'Train_phase2/latest';
SPECIFIC_DIRS.PrepDataset = 'PrepDataset_phase3';
SPECIFIC_DIRS.Train = 'Train_phase3';
SPECIFIC_DIRS.TrainSnapshot = 'Train_phase3/latest';

CACHE_DIR = fullfile( fileparts(mfilename('fullpath')), '../cache' ); 
SPECIFIC_DIRS = structfun(@(sf) fullfile(CACHE_DIR,sf), SPECIFIC_DIRS, 'UniformOutput', 0);

SPECIFIC_DIRS.depend_InitSnapshot = PRE_PARAM.SPECIFIC_DIRS;
