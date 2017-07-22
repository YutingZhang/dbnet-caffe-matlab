
PARAM = sysReadParam( STAGE_SPECIFIC_DIR );

PARAM.depend_InitSnapshot = sysStageMixedParam('InitSnapshot',[],0,PARAM,0,1);
PARAM.Train_Finetune = 1;

if isfield(PARAM, 'ConvFeatNet_LR') && ~isempty(PARAM.ConvFeatNet_LR)
    PARAM.ConvFeatNet_LR = PARAM.ConvFeatNet_LR / 10;
end

if isfield(PARAM, 'RegionProposalNet_LR') && ~isempty(PARAM.RegionProposalNet_LR)
    PARAM.RegionProposalNet_LR = PARAM.RegionProposalNet_LR / 10;
end

if isfield(PARAM, 'RegionFeatNet_LR') && ~isempty(PARAM.RegionFeatNet_LR)
    PARAM.RegionFeatNet_LR = PARAM.RegionFeatNet_LR / 10;
end

if isfield(PARAM, 'TextFeatNet_LR') && ~isempty(PARAM.TextFeatNet_LR)
    PARAM.TextFeatNet_LR = PARAM.TextFeatNet_LR / 10;
end

if isfield(PARAM, 'PairNet_LR') && ~isempty(PARAM.PairNet_LR)
    PARAM.PairNet_LR = PARAM.PairNet_LR / 10;
end
