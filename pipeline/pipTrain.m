% Pipline stage: Train

%% Parameters
% define parameters that do not affect output here
% e.g. OTHER_DEFAULT_PARAM.A = 0
OTHER_DEFAULT_PARAM = struct();

OTHER_DEFAULT_PARAM.Train_SnapshotFrequency = 20000;

% snapshot at particular iterations
OTHER_DEFAULT_PARAM.Train_SnapshotAt        = [];
OTHER_DEFAULT_PARAM.Train_SnapshotOnInterruption = 0;
OTHER_DEFAULT_PARAM.Train_MaxIteration      = inf;

% use the latest checkpoint at or before Train_StartIteration 
OTHER_DEFAULT_PARAM.Train_StartIteration    = inf; 

OTHER_DEFAULT_PARAM.Use_DeviceMat = 1;

OTHER_DEFAULT_PARAM.Dump_SampleVisualization  = 0;

OTHER_DEFAULT_PARAM.GPU_ID = str2num(getenv('GPU_ID')); % [] for no GPU

PARAM = xmerge_struct( OTHER_DEFAULT_PARAM, PARAM );



%% initialization for stage


%% Start your code here

%% Initialization

% set up GPU
assert( ~isempty(PARAM.GPU_ID), 'must use GPU, CPU mode not supported' );
matcaffeSetGPU( PARAM.GPU_ID );

use_device_mat = PARAM.Use_DeviceMat && ~isempty(PARAM.GPU_ID);

% load ConvFeatNet
solver_struct = 'noupdate';
if PARAM.ConvFeatNet_Updatable
    solver_struct = nldetMergeSolverAndLR( PARAM.ConvFeatNet_Solver, PARAM.ConvFeatNet_LR );
end
cfn = nldetLoadBlock( SPECIFIC_DIRS.ConvFeatNet, ...
    PARAM.ConvFeatNet_BatchSize, ...
    solver_struct, ...
    PARAM.ConvFeatNet_Param ...
    );
nldetBlockUseDeviceMat(cfn, use_device_mat);

% load RegionProposalNet
solver_struct = 'noupdate';
if PARAM.RegionProposalNet_Updatable
    solver_struct = nldetMergeSolverAndLR( PARAM.RegionProposalNet_Solver, PARAM.RegionProposalNet_LR );
end
% RegionProposalNet_Param1 = PARAM.RegionProposalNet_Param;
% RegionProposalNet_Param1.Maximum_BoxNum = ...
%     PARAM.Maximum_ProposedBoxNum;
% RegionProposalNet_Param1.ConvFeatNet_Block = cfn; % only useful for dependent block
rpn = nldetLoadBlock( SPECIFIC_DIRS.RegionProposalNet, ...
    PARAM.RegionProposalNet_BatchSize, ...
    solver_struct, ...
    PARAM.RegionProposalNet_Param, ...
    scalar_struct('input_force_backward',{1}) ...
    );
nldetBlockUseDeviceMat(rpn, use_device_mat);

% load RegionFeatNet
solver_struct = 'noupdate';
if PARAM.RegionFeatNet_Updatable
    solver_struct = nldetMergeSolverAndLR( PARAM.RegionFeatNet_Solver, PARAM.RegionFeatNet_LR );
end
rfn = nldetLoadBlock( SPECIFIC_DIRS.RegionFeatNet, ...
    [PARAM.BaseModel_BatchSize, PARAM.RegionFeatNet_BatchSize], ...
    solver_struct, ...
    PARAM.RegionFeatNet_Param, ...
    scalar_struct('input_force_backward',{1}) ...
    );
nldetBlockUseDeviceMat(rfn, use_device_mat);

% load TextFeatNet
solver_struct = 'noupdate';
if PARAM.TextFeatNet_Updatable
    solver_struct = nldetMergeSolverAndLR( PARAM.TextFeatNet_Solver, PARAM.TextFeatNet_LR );
end
tfn = nldetLoadBlock( SPECIFIC_DIRS.TextFeatNet, ...
    PARAM.TextFeatNet_BatchSize, ...
    solver_struct, ...
    PARAM.TextFeatNet_Param, ...
    scalar_struct('input_force_backward',{1}) ...
    );
nldetBlockUseDeviceMat(tfn, use_device_mat);

% load PairNet
solver_struct = 'noupdate';
if PARAM.PairNet_Updatable
    solver_struct = nldetMergeSolverAndLR( PARAM.PairNet_Solver, PARAM.PairNet_LR );
end
prn = nldetLoadBlock( SPECIFIC_DIRS.PairNet, ...
    PARAM.PairNet_BatchSize, ...
    solver_struct, ...
    PARAM.PairNet_Param, ...
    scalar_struct('input_force_backward', {1,2}) ...
    );
nldetBlockUseDeviceMat(prn, use_device_mat);

% init batch sampler

data_loader = nldetGetDataLoaderConstructor( SPECIFIC_DIRS.PrepDataset );

im_sampler_param = struct();
im_sampler_param.shuffle = 1;
im_sampler_param.limit   = PARAM.Train_ImageLimit;
im_sampler_param.prefetch_size = PARAM.ConvFeatNet_BatchSize*4;

im_sampler = data_loader.image_sampler( PARAM.Train_DataSubset, ...
    im_sampler_param, SPECIFIC_DIRS.PrepDataset );
gt_loader  = data_loader.gt_loader( PARAM.Train_DataSubset, ...
    SPECIFIC_DIRS.PrepDataset );

nbs_param = struct();
nbs_param.gt_nearby_threshold = PARAM.Sampler_GTNearby_Threshold;
if isempty( PARAM.Sampler_LabelRemapping )
    nbs_param.score_remapping_func_str  = [];
    nbs_param.score_remapping_func_args = {};
else
    switch PARAM.Sampler_LabelRemapping
        case 'threshold'
            nbs_param.score_remapping_func_str  = 'threshold_detection_score';
            nbs_param.score_remapping_func_args = { ...
                PARAM.Sampler_LabelThreshold_Negative , ...
                PARAM.Sampler_LabelThreshold_Positive };
        case 'power'
            nbs_param.score_remapping_func_str  = 'power_detection_score';
            nbs_param.score_remapping_func_args = { ...
                PARAM.Sampler_LabelThreshold_Negative , ...
                PARAM.Sampler_LabelThreshold_Positive , ...
                PARAM.Sampler_LabelReshape_Power };
        otherwise
            error( 'Unknown Sampler_LabelRemapping' );
    end
end
nbs_param.same_image_gt_as_neg = PARAM.Sampler_NegativeTextFromSameImage;
nbs_param.text_compatibility_threshold = PARAM.Sampler_TextSimilarityThreshold;

nbs = nldetBatchSampler(PARAM.BaseModel_BatchSize,im_sampler,gt_loader,nbs_param);
if PARAM.Sampler_RandomNegativeTextNum > 0
    nbs.set_random_choser( PARAM.Sampler_RandomNegativeTextNum );
end
if PARAM.Sampler_Image_HardNegativeNum > 0 || ...
        PARAM.Sampler_Region_HardNegativeNum > 0
    hard_miner_param = struct();
    hard_miner_param.image_chosen   = PARAM.Sampler_Image_HardNegativeNum;
    hard_miner_param.region_chosen  = PARAM.Sampler_Region_HardNegativeNum;
    hard_miner_param.image_chached  = hard_miner_param.image_chosen  * 2;
    hard_miner_param.region_chached = hard_miner_param.region_chosen * 2;
    hard_miner_param.hard_threshold = PARAM.Sampler_HardNegative_Threshold;
    nbs.set_hard_miner( hard_miner_param );
end

% initialize pipeline

ibs = nldetBatchStacker( SPECIFIC_DIRS.BatchStacker );

npl_param = struct();

npl_param.region_proposal_top_num    = PARAM.Sampler_ProposalTopNum;
npl_param.region_proposal_random_num = PARAM.Sampler_ProposalRadomNum;

if PARAM.Dump_SampleVisualization
    npl_param.visualization_path = STAGE_SPECIFIC_DIR;
end
npl_param.region_proposal_input = nldetGetRegionProposalInputType( ...
    SPECIFIC_DIRS.RegionProposalNet );

npl = nldetPipeline( 'train', nbs, ibs, cfn, rpn, rfn, tfn, prn, npl_param );

npl.toggle_block_updatable( 'conv_feat_net',       PARAM.ConvFeatNet_Updatable );
npl.toggle_block_updatable( 'region_proposal_net', PARAM.RegionProposalNet_Updatable );
npl.toggle_block_updatable( 'region_feat_net',     PARAM.RegionFeatNet_Updatable );
npl.toggle_block_updatable( 'text_feat_net',       PARAM.TextFeatNet_Updatable );
npl.toggle_block_updatable( 'pair_net',            PARAM.PairNet_Updatable );

if PARAM.Train_Finetune
    ftSD = SPECIFIC_DIRS.depend_InitSnapshot;
    npl.toggle_block_restore_solver( 'batch_sampler', ...
        strcmp(SPECIFIC_DIRS.PrepDataset, ftSD.PrepDataset) );
    npl.toggle_block_restore_solver( 'conv_feat_net', ...
        strcmp(SPECIFIC_DIRS.ConvFeatNet, ftSD.ConvFeatNet) );
    npl.toggle_block_restore_solver( 'region_proposal_net', ...
        strcmp(SPECIFIC_DIRS.RegionProposalNet, ftSD.RegionProposalNet) );
    npl.toggle_block_restore_solver( 'region_feat_net', ...
        strcmp(SPECIFIC_DIRS.RegionFeatNet, ftSD.RegionFeatNet) );
    npl.toggle_block_restore_solver( 'text_feat_net', ...
        strcmp(SPECIFIC_DIRS.TextFeatNet, ftSD.TextFeatNet) );
    npl.toggle_block_restore_solver( 'pair_net', ...
        strcmp(SPECIFIC_DIRS.PairNet, ftSD.PairNet) );
end

%% Restore and Train
nldetTrain_Script_Restore
npl_resume = @nldetTrain_Script_Run;
nldetTrain_Script_Run
