

nldetParamReduction_FullNet

% solver struct
%  empty when no caffe net
% learning rate
%  empty when no caffe net

BaseModel_Solver = nldetCanonicalizeBlockSolverStruct( ...
    BaseModel_Solver, [], struct());

if ConvFeatNet_Updatable
    ConvFeatNet_Solver = nldetCanonicalizeBlockSolverStruct( ...
        ConvFeatNet_Solver, BaseModel_Solver, ConvFeatNet_HasCaffe);
    ConvFeatNet_LR = nldetCanonicalizeBlockScalar( ...
        ConvFeatNet_LR, BaseModel_LR, ConvFeatNet_HasCaffe);
end

if RegionProposalNet_Updatable
    RegionProposalNet_Solver = nldetCanonicalizeBlockSolverStruct( ...
        RegionProposalNet_Solver, BaseModel_Solver, RegionProposalNet_HasCaffe);
    RegionProposalNet_LR = nldetCanonicalizeBlockScalar( ...
        RegionProposalNet_LR, BaseModel_LR, RegionProposalNet_HasCaffe);
end

if RegionFeatNet_Updatable
    RegionFeatNet_Solver = nldetCanonicalizeBlockSolverStruct( ...
        RegionFeatNet_Solver, BaseModel_Solver, RegionFeatNet_HasCaffe);
    RegionFeatNet_LR = nldetCanonicalizeBlockScalar( ...
        RegionFeatNet_LR, BaseModel_LR, RegionFeatNet_HasCaffe);
end

if TextFeatNet_Updatable
    TextFeatNet_Solver = nldetCanonicalizeBlockSolverStruct( ...
        TextFeatNet_Solver, BaseModel_Solver, TextFeatNet_HasCaffe);
    TextFeatNet_LR = nldetCanonicalizeBlockScalar( ...
        TextFeatNet_LR, BaseModel_LR, TextFeatNet_HasCaffe);
end

if PairNet_Updatable
    PairNet_Solver = nldetCanonicalizeBlockSolverStruct( ...
        PairNet_Solver, BaseModel_Solver, PairNet_HasCaffe);
    PairNet_LR = nldetCanonicalizeBlockScalar( ...
        PairNet_LR, BaseModel_LR, PairNet_HasCaffe);
end

;
