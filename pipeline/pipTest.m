% Pipline stage: Test

%% Parameters
% define parameters that do not affect output here
% e.g. OTHER_DEFAULT_PARAM.A = 0
OTHER_DEFAULT_PARAM = struct();
OTHER_DEFAULT_PARAM.GPU_ID = str2num(getenv('GPU_ID'));
OTHER_DEFAULT_PARAM.FORCE  = 0;

OTHER_DEFAULT_PARAM.Use_DeviceMat = 1;

PARAM = xmerge_struct( OTHER_DEFAULT_PARAM, PARAM );



%% initialization for stage


%% Start your code here

if PARAM.Test_External
    fprintf('Please put the external test results as results.v7.mat by yourself.\n');
    return;
end

% set up GPU
assert( ~isempty(PARAM.GPU_ID), 'must use GPU, CPU mode not supported' );
matcaffeSetGPU( PARAM.GPU_ID );

use_device_mat = PARAM.Use_DeviceMat && ~isempty(PARAM.GPU_ID);

% load ConvFeatNet
cfn = nldetLoadBlock( SPECIFIC_DIRS.ConvFeatNet, ...
    PARAM.ConvFeatNet_BatchSize, [], ...
    PARAM.ConvFeatNet_Param ...
    );
nldetBlockUseDeviceMat(cfn, use_device_mat);

% load RegionProposalNet
% RegionProposalNet_Param1 = PARAM.RegionProposalNet_Param;
% RegionProposalNet_Param1.Maximum_BoxNum = ...
%     PARAM.Maximum_ProposedBoxNum;
% RegionProposalNet_Param1.ConvFeatNet_Block = cfn; % only useful for dependent block
rpn = nldetLoadBlock( SPECIFIC_DIRS.RegionProposalNet, ...
    PARAM.RegionProposalNet_BatchSize, [], ...
    PARAM.RegionProposalNet_Param );
nldetBlockUseDeviceMat(rpn, use_device_mat);

% load RegionFeatNet
rfn = nldetLoadBlock( SPECIFIC_DIRS.RegionFeatNet, ...
    [PARAM.BaseModel_BatchSize, PARAM.RegionFeatNet_BatchSize], [], ...
    PARAM.RegionFeatNet_Param );
nldetBlockUseDeviceMat(rfn, use_device_mat);

% load TextFeatNet
tfn = nldetLoadBlock( SPECIFIC_DIRS.TextFeatNet, ...
    PARAM.TextFeatNet_BatchSize, [], ...
    PARAM.TextFeatNet_Param );
nldetBlockUseDeviceMat(tfn, use_device_mat);

% load PairNet
prn = nldetLoadBlock( SPECIFIC_DIRS.PairNet, ...
    PARAM.PairNet_BatchSize, [], ...
    PARAM.PairNet_Param );
nldetBlockUseDeviceMat(prn, use_device_mat);

% load BatchSampler

INDEX_LOCK_PATH = fullfile( STAGE_SPECIFIC_DIR, '_index.lock' );

data_loader = nldetGetDataLoaderConstructor( SPECIFIC_DIRS.PrepDataset );

im_sampler_param = struct();
im_sampler_param.shuffle = 0;
im_sampler_param.limit   = PARAM.Test_ImageLimit;
im_sampler_param.unique_guard = INDEX_LOCK_PATH;
im_sampler_param.prefetch_size = PARAM.ConvFeatNet_BatchSize*4;

im_sampler = data_loader.image_sampler( PARAM.Test_DataSubset, ...
    im_sampler_param, SPECIFIC_DIRS.PrepDataset );
gt_loader  = data_loader.gt_loader( PARAM.Test_DataSubset, ...
    SPECIFIC_DIRS.PrepDataset );

nbs_param = struct();
nbs_param.same_image_gt_as_neg = ~PARAM.Test_WithPredefinedPhrases;
nbs_param.gt_nearby_threshold = 1;
nbs_param.score_remapping_func_str  = [];
nbs_param.score_remapping_func_args = {};
nbs_param.conflicting_policy = 'scoring'; % 'avoiding' for train

nbs=nldetBatchSampler(PARAM.BaseModel_BatchSize,im_sampler,gt_loader,nbs_param);


npl_param = struct();

if PARAM.Test_WithPredefinedPhrases
    ppn = nldetLoadBlock( SPECIFIC_DIRS.PredefinedPhraseNet, ...
        [], [], ...
        PARAM.PredefinedPhraseNet_Param );
    nta_pp = nldetTextAugmenter_PredefinedPhrases(ppn);
    nbs.add_text_augmenter( 'PredefinedPhrases', nta_pp );
    npl_param.remove_orphan_phrases_in_results = true;
end

% init pipeline object

ibs = nldetBatchStacker( SPECIFIC_DIRS.BatchStacker );

npl_param.region_proposal_input = nldetGetRegionProposalInputType( ...
    SPECIFIC_DIRS.RegionProposalNet );

npl = nldetPipeline( 'test', nbs, ibs, cfn, rpn, rfn, tfn, prn, npl_param );

%% Load model

finetune_point = SPECIFIC_DIRS.TrainSnapshot;
t1 = tic_print( 'Test on : %s\n', finetune_point );
npl.load_pretrained( finetune_point );
toc_print( t1 );

%% Test

RESULTS_FN = fullfile(STAGE_SPECIFIC_DIR, 'results.v7.mat' );

numImages = im_sampler.count();

if exist( RESULTS_FN, 'file' )
    if ~PARAM.FORCE
        fprintf( 'Results file already exists\n' );
        return;
    else
        im_sampler.atomic_indexer.reset();
        movefile( RESULTS_FN, [RESULTS_FN '.bak'] );
    end
end

result_storage = sliced_storage( RESULTS_FN, numImages, 1 );
im_sampler.atomic_indexer.set_predone_func( @(n) result_storage.slice_exists(n) );

batch_index = 0;
while nbs.processing_level < nldetBatchSampler.PLEVEL_INVALID
    batch_index = batch_index + 1;
    t1 = tic_print( 'Batch %d : ', batch_index );
    npl.step();
    cur_iters = nbs.iters_in_epoch;
    fprintf( ' %s / %d : ', vec2str(cur_iters), numImages );
    for j=1:numel(cur_iters)
        R = npl.formatted_test_results(j);
        result_storage.set_slice(cur_iters(j),R);
    end
    toc_print(t1);
end

result_storage.fuse();


