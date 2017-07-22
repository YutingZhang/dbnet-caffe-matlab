function [X,varargout] = caffeproto_replace_func_upgrade( varargin )

if strcmp(varargin{1},'extend')
    X = {};
    return;
elseif strcmp(varargin{1},'adjacent')
    X = [0];
    return;
end

X = [];

subS = varargin{1};
if ~strcmp( subS.type{1}, 'ConvCombo' ), return; end
if ~strcmp( subS.conv_combo_param.type, 'Deconvolution' ), return; end

Pdef = struct();
Pdef.poolAveCoeffLearnable = false;
Pdef.poolAveScale  = 0;
Pdef.poolAveWeight = 0;
Pdef.poolAveWeightLRmult = 1e4;

Pdef.fida_conv_param.pre_scaled      = true;
Pdef.fida_conv_param.mean_removal    = false;
Pdef.fida_conv_param.back_scaling_up = true;

Pdef.unpoolMask = true;


PARAM = scalar_struct(varargin{end}{:});
PARAM = xmerge_struct('always','always', Pdef, PARAM);

X = subS;
X.conv_combo_param.type = {'FidaDeconv'};
X.fida_conv_param = PARAM.fida_conv_param;
has_unpooling = ismember( 'Unpooling', X.conv_combo_param.aux_layers );

if has_unpooling

    if abs(PARAM.poolAveScale)<eps 
        X.pooling_param.bias_weight  = PARAM.poolAveWeight;
        X.fida_conv_param.pre_scaled = false;
        if X.pooling_param.kernel_size>X.pooling_param.stride
            X.pooling_param.array_normalized = [true];
        end
    else
        X.pooling_param.pool(2) = pbEnum('AVE');
        X.pooling_param.mult_scales  = [1,PARAM.poolAveScale];
        X.pooling_param.mult_weights = [1,PARAM.poolAveWeight*X.pooling_param.kernel_size.^2];
    end
    if PARAM.poolAveCoeffLearnable
        X.pooling_param.weight_learnable = true;
        X.pooling_param.weight_lr_mult   = PARAM.poolAveWeightLRmult;
    end

    if PARAM.unpoolMask
        X = caffeproto_convcombo_set_aux_blobs( ...
            X, 'Unpooling', [], {[X.bottom{1} '/unpool-mask']} );
        X = caffeproto_convcombo_set_aux_blobs( ...
            X, 'ConvCombo', {[X.bottom{1} '/unpool-mask']}, [] );
        X.propagate_down = [true false]; % hard coded (Fida Layer doesn't backpropagate fidelity)
    end

end
