function [X, varargout] = caffeproto_replace_func_cae_expand( varargin )

ARGS = varargin{end};

if ischar(varargin{1})
    if strcmp(varargin{1},'extend')
        X = { [{@(varargin) caffeproto_replace_func_cae_expand( 'expand-init', varargin{:} )}, ARGS], ...
            { 'list', 'iterative', ... 
            [ {@(varargin) caffeproto_replace_func_cae_expand( 'expand-step', varargin{:} )}, ARGS ] }, ...
            };
        return;
    end
end

VAR_IN = varargin(2:end);

if ~isempty(VAR_IN) && strcmp( VAR_IN{1}, 'extend' )
    X = {};
    return;
end

Enc2DecTypes = {
    'Convolution',  'Deconvolution', 'FidaDeconv'
    'Deconvoltion', 'Convolution',   'FidaConv'
    'FidaConv',     'FidaDeconv',    'FidaDeconv'
    'FidaDeconv',   'FidaConv',      'FidaConv'
    'InnerProduct', 'InnerProduct',  'InnerProduct'
    'Eltwise',      'Eltwise',       'Eltwise'
    'Pooling',      'Deconvolution', 'Deconvolution' };
P0.useFida = 0; % 0 no, 1 use for depooling, 2-always, -1 - not for pooling

P0.enlargeKernel = 0;

P0.decoderPrefix = 'dec:';

% P0.encoderZeroLR = 1;
P0.unpoolingMask = 'fix'; % 'fix', 'ave', 'known', 'pred:hard', 'pred:soft'
P0.unpoolingMethod_Unknown = 'fix'; % 'fix', 'ave'
P0.switchPredictionK = [3, 3];      % kernel size
P0.switchPredictionC = [2, inf];    % mult of channel number
%P0.switchPredictionGstd =  'xavier'; %0.001;    % std for weight filler
P0.switchPredictionGstd =  0.001;    % std for weight filler
P0.backpropSoftSwitch = 0;  % back-propagation through switch
P0.poolAveWeight = 0;
P0.poolAveScale  = 0;
P0.poolAveCoeffLearnable = 0;
P0.poolAveWeightLRmult = 1e4;

% '@auto' (figure the blob name automatically), 
%  otherwise it is the specified blob name.
P0.trainWithReconRef    = '@auto'; 
P0.trainReconLossWeight = 0;
P0.trainRefScale = 1;

P0.lrMultFunc = @(n) 1;

P0.trainInterReconLossWeight = -1;

P0.trainSwitchLossWeight = 0;

P0.dumpImage   = 0;
P0.dumpMeanRef = 'mean';

P0.unpoolNormalization = true; % true - auto determine, false - disable

P0.fida_conv_param.pre_scaled      = true;
P0.fida_conv_param.mean_removal    = false;
P0.fida_conv_param.back_scaling_up = true;

P0.ignoreLastPooling = false;

P0.weight_filler = [];

P0.keepLRN     = 0;
P0.keepDropout = 0;

% 'decoder' - link to the lower layer of the decoder
% 'encoder' - link reset to encoder
% 'ladder'  - ladder
P0.linkTypeAtLoss = 'decoder';

P0.enableNonprefixable = 0;

%
P0.addNoise2Encoder = 0;
P0.encoderNoiseAdaptive = 0;

if isstruct(ARGS)
    P1 = ARGS;
else
    ARGS(2:2:end) = cellfun( @(a) {a},ARGS(2:2:end), 'UniformOutput', 0 );
    P1 = struct( ARGS{:} );
end
PARAM = xmerge_struct( 'always', 'always', P0, ...
    partial_struct(P1,fieldnames(P0)) );

assert( ismember( PARAM.linkTypeAtLoss, {'decoder','encoder','ladder'} ), ...
    'unrecognized linkTypeAtLoss' );

if ischar(PARAM.useFida)
    switch PARAM.useFida
        case 'no'
            PARAM.useFida = 0;
        case 'pool'
            PARAM.useFida = 1;
        case 'always'
            PARAM.useFida = 2;
        case 'nopool'
            PARAM.useFida = -1;
        otherwise
            error( 'Unrecognized tag for useFida' );
    end
end

switch varargin{1}
    case 'expand-init'
        if strcmp( VAR_IN{1}, 'adjacent' )
            X = [0];
            return;
        end
        X = [];
        subS = VAR_IN{1};
        if ~strcmp( subS.type, 'CAEStack' ), return; end
        X = subS;
        X.cae_stack_param.step = 0;
        X.cae_stack_param.added_loss = {};
        X.cae_stack_param.propagate_top = [];
        X.cae_stack_param.pre_pool_stride = 1;
        % add training loss
    case 'expand-step'
        if strcmp( VAR_IN{1}, 'adjacent' )
            X = [0];
            return;
        end
        X = [];
        subS = VAR_IN{1};
        if ~strcmp( subS.type, 'CAEStack' ), return; end
        if isempty(subS.layers)
            sp= struct([]);
            if ~isempty(subS.top)
                sp = struct();
                sp.name = {[ PARAM.decoderPrefix 'split:' subS.bottom{1} ]};
                sp.type = {'Split:Branch'};
                sp.bottom = subS.bottom;
                sp.top    = subS.top;
            end
            X = struct(); % just a rename layer
            X.name = {[ PARAM.decoderPrefix 'connect:' subS.bottom{1} ]};
            X.type = {'Reshape'};
            X.bottom = subS.bottom;
            X.reshape_param.num_axes = 0;
            X.top    = { [PARAM.decoderPrefix subS.bottom{1}] };
            X = cat_struct(2,sp,X);
        else
            p = subS.layers(end);
            t = struct([]); l = struct([]); 
            lc = struct([]); % ladder combinator
            splc = struct([]); % split connector
            ts = struct([]);
            X = subS;
            X.layers(end) = [];
            X.bottom = p.top;
            X.cae_stack_param.step = X.cae_stack_param.step + 1;
            if strcmp(p.type{1}, 'ConvCombo')
                has_pooling0 = ismember ( 'Pooling' , p.conv_combo_param.aux_layers);
                if isempty(X.layers)
                    if PARAM.ignoreLastPooling
                        if ~isempty(p.conv_combo_param.aux_layers)
                            p.conv_combo_param.aux_layers = ...
                                setdiff(p.conv_combo_param.aux_layers, {'Pooling'}, 'stable' );
                        end
                    end
                end
                
                inCh = subS.cae_stack_param.input_channels;
                X.cae_stack_param.input_channels = [];
                encType = p.conv_combo_param.type{1};
                [~,layerTypeIdx] = ismember( encType, Enc2DecTypes(:,1) );
                assert( layerTypeIdx>0, 'Unknown ConvCombo internal type' );

                assert( ~isempty(inCh), ...
                    'Cannot proceed without knowing the input channel number' );
                t = partial_struct(p, '@exclude', 'aux');
                if isfield( t, 'param' ) && isfield( t.param, 'lr_mult_zero' )
                    t.param = rmfield(t.param,'lr_mult_zero');
                end
                t.conv_combo_param = partial_struct(t.conv_combo_param, '@exclude', 'sideloss' );
                
                if isfield( t.conv_combo_param, 'aux_layernames' );
                    t.conv_combo_param = rmfield( t.conv_combo_param, 'aux_layernames' );
                end
                
                if ~PARAM.keepLRN
                    t.conv_combo_param.aux_layers = setdiff( t.conv_combo_param.aux_layers, {'LRN'}, 'stable' );
                end
                if ~PARAM.keepDropout
                    t.conv_combo_param.aux_layers = setdiff( t.conv_combo_param.aux_layers, {'Dropout'}, 'stable' );
                end

                
                if PARAM.useFida>=2 || PARAM.useFida<0 || (PARAM.useFida && strcmp( p.type, 'ConvCombo' ) && ...
                        ismember( 'Pooling', p.conv_combo_param.aux_layers ))
                    decType = Enc2DecTypes{layerTypeIdx,3};
                else
                    decType = Enc2DecTypes{layerTypeIdx,2};
                end
                
                if isfield( p, 'param' ) && ~isempty( p.param )
                    lr_mult_mult = PARAM.lrMultFunc(X.cae_stack_param.step);
                    for k = 1:length(p.param)
                        if isfield( p.param(k), 'lr_mult' ) && ~isempty(p.param(k).lr_mult)
                            t.param(k).lr_mult = p.param(k).lr_mult * lr_mult_mult;
                        end
                    end
                end
                
                X.cae_stack_param.input_channels = caffeproto_avail_ioshape( ...
                    p, 't', 1 );
                if isempty(X.cae_stack_param.input_channels)
                    if strcmp( decType, 'InnerProduct' )
                        X.cae_stack_param.input_channels = p.inner_product_param.num_output;
                    elseif strcmp( decType, 'Eltwise' )
                        % do nothing
                    elseif strcmp( encType, 'Pooling' )
                        % do nothing
                    else
                        X.cae_stack_param.input_channels = p.convolution_param.num_output;
                    end
                end
                t.conv_combo_param.input_channels = X.cae_stack_param.input_channels;
                p.conv_combo_param.input_channels = inCh;

                
                isFida = strncmp( decType, 'Fida', 4 );
                
                t.conv_combo_param.type = {decType};
                t.name = {[PARAM.decoderPrefix p.name{1}]};
                t.top = {[PARAM.decoderPrefix p.bottom{:}]};
                %if isempty(X.layers)
                %    t.bottom = p.top;
                %else
                t.bottom = {[PARAM.decoderPrefix p.top{:}]};
                %end
                
                pre_pool_stride = 1;
                if strcmp( decType, 'InnerProduct' )
                    if length(inCh)>=2
                        t.inner_product_param.num_output = prod(inCh);
                        % add a reshape layer
                        t.conv_combo_param.aux_layers = union( ...
                            t.conv_combo_param.aux_layers, 'Reshape', 'stable' );
                        t.reshape_param.axis = 1;
                        t.reshape_param.shape.dim = inCh;
                    else
                        t.inner_product_param.num_output = inCh;
                    end
                    if ~isempty( PARAM.weight_filler )
                        t.inner_product_param.weight_filler = PARAM.weight_filler; 
                    end
                elseif strcmp( decType, 'Eltwsie' )
                    % do nothing
                    pre_pool_stride = X.cae_stack_param.pre_pool_stride; % keep pre_pool_stride
                elseif strcmp( encType, 'Pooling' )
                    t.convolution_param = partial_struct( p.pooling_param, ...
                        '@exclude', 'global_pooling', 'pool' );
                    if default_eval( 'p.pooling_param.global_pooling', false )
                        unpooling_size = caffeproto_avail_ioshape( p, 'b', 1 );
                        assert( ~isempty(unpooling_size), 'No shape information found' );
                        unpooling_size = unpooling_size(2:end);
                        if all(unpooling_size(1)==unpooling_size), 
                            unpooling_size = unpooling_size(1);
                        end
                        t.convolution_param.kernel_size = unpooling_size; %??
                    end
                    pre_pool_stride = default_eval( 'p.pooling_param.stride', [] );
                    if isempty(pre_pool_stride)
                        pre_pool_stride_h  = default_eval( 'p.pooling_param.stride_h', 1 );
                        pre_pool_stride_w  = default_eval( 'p.pooling_param.stride_w', 1 );
                        if pre_pool_stride_h == pre_pool_stride_w
                            pre_pool_stride = pre_pool_stride_h;
                        else
                            pre_pool_stride = [pre_pool_stride_h, pre_pool_stride_w];
                        end
                    end
                else % convolution
                    t.convolution_param.num_output = inCh(1);
                    t.convolution_param.group = [];
                    if isFida 
                        if ~isfield( t, 'fida_conv_param' )
                            t.fida_conv_param = PARAM.fida_conv_param;
                        end
                        if isfield(t.fida_conv_param,'pre_scaled') && ...
                                ~t.fida_conv_param.pre_scaled
                            warning( 'fida_conv_param.pre_scaled is forced to be true' );
                            t.fida_conv_param.pre_scaled = true;
                        end
                    end
                    if ~isempty( PARAM.weight_filler )
                        t.convolution_param.weight_filler = PARAM.weight_filler; 
                    end
                    % if kernel_size < stride, then enlarge kernel
                    conv_kernel = default_eval( 't.convolution_param.kernel_size', 1 );
                    conv_stride = default_eval( 't.convolution_param.stride', 1 );
                    conv_pre_pool_stride = conv_stride.*X.cae_stack_param.pre_pool_stride;
                    t.convolution_param.kernel_size = max(conv_kernel,conv_pre_pool_stride);
                end
                X.cae_stack_param.pre_pool_stride = pre_pool_stride;
                
                [~,aux_pool_idx] = ismember( 'Pooling', t.conv_combo_param.aux_layers );
                if aux_pool_idx > 0
                    t.conv_combo_param.aux_layers{aux_pool_idx} = 'Unpooling';
                    poolingMethod = p.pooling_param.pool.val();
                    unpoolingMethod = PARAM.unpoolingMask;
                    if ismember( poolingMethod, {'FIX','AVE'} )
                        unpoolingMethod = PARAM.unpoolingMethod_Unknown;
                    end
                    if ismember(unpoolingMethod,{'ave','fix'})
                        switch unpoolingMethod
                            case 'fix'
                                t.pooling_param.pool = pbEnum( 'FIX' );
                            case 'ave'
                                t.pooling_param.pool = pbEnum( 'AVE' );
                            otherwise
                                error( 'Unknown unpooling mask type (unpoolingMask)' );
                        end
                    else
                        assert( ismember( p.pooling_param.pool.val(), {'MAX'} ) );
                        switch unpoolingMethod
                            case 'known'
                                p = caffeproto_convcombo_set_aux_blobs( ...
                                    p, 'Pooling', [], {[p.top{1} '/pool-mask']} );
                                t = caffeproto_convcombo_set_aux_blobs( ...
                                    t, 'Unpooling', {[p.top{1} '/pool-mask']}, [] );
                                t.pooling_param.pool = pbEnum( 'SWITCH' );
                                t.conv_combo_param.unpooling_backprop_switch = false;
                                % p.pooling_param.mask_index_type = pbEnum( 'GLOBAL' ); % fast than LOCAL
                                % t.pooling_param.mask_index_type = pbEnum( 'GLOBAL' );
                            case {'pred:hard','pred:soft'}
                                % set up params and blobs
                                if strcmp(unpoolingMethod,'pred:hard')
                                    t.pooling_param.pool = pbEnum( 'SWITCH' );
                                    p.pooling_param.mask_index_type = pbEnum( 'LOCAL' );
                                    t.pooling_param.mask_index_type = pbEnum( 'LOCAL' );
                                    t = caffeproto_convcombo_set_aux_blobs( ...
                                        t, 'Unpooling', {[t.bottom{1} '/pool-pred/hard']}, [] );
                                    t.conv_combo_param.unpooling_backprop_switch = false;
                                else
                                    t.pooling_param.pool = pbEnum( 'SOFT_SWITCH' );
                                    t = caffeproto_convcombo_set_aux_blobs( ...
                                        t, 'Unpooling', {[t.bottom{1} '/pool-pred/soft']}, [] );
                                    t.conv_combo_param.unpooling_backprop_switch = ...
                                        logical(PARAM.backpropSoftSwitch);
                                end
                                % add prediction path
                                predLayerN = numel(PARAM.switchPredictionK);
                                assert( numel( PARAM.switchPredictionC ) == predLayerN, ...
                                    'switchPredictionK should have the same number of elements' );
                                if ischar( PARAM.switchPredictionGstd )
                                    PARAM.switchPredictionGstd = {PARAM.switchPredictionGstd};
                                end
                                if isscalar( PARAM.switchPredictionGstd )
                                    PARAM.switchPredictionGstd = repmat( ...
                                        PARAM.switchPredictionGstd, size(PARAM.switchPredictionK) );
                                end
                                assert( numel( PARAM.switchPredictionGstd ) == predLayerN, ...
                                    'switchPredictionGstd should have the same number of elements or be a scalar' );
                                poolK = p.pooling_param.kernel_size.^2;
                                %preTopName = [t.bottom{1} '/pool'];
                                preTopName = t.bottom{1};
                                for j = 1:predLayerN
                                    tsl = struct();
                                    tsl.name = {[t.bottom{1} '/pool-pred/pre-' int2str(j)]};
                                    tsl.type = {'Convolution'};
                                    tsl.bottom = { preTopName };
                                    tsl.top = { [t.bottom{1} '/pool-pred/pre-' int2str(j)] };
                                    tsl.convolution_param.num_output  = ...
                                        X.cae_stack_param.input_channels(1) * ...
                                        min( PARAM.switchPredictionC(j), poolK );
                                    tsl.convolution_param.kernel_size = ...
                                        floor(PARAM.switchPredictionK(j)/2)*2+1;
                                    tsl.convolution_param.pad = ...
                                        floor(PARAM.switchPredictionK(j)/2);
                                    tsl.convolution_param.stride = 1;
                                    if isnumeric(PARAM.switchPredictionGstd)
                                        tsl.convolution_param.weight_filler = ...
                                            struct('type', {{'gaussian'}}, ...
                                            'std', {PARAM.switchPredictionGstd(j)});
                                    elseif isstruct( PARAM.switchPredictionGstd{j} )
                                        tsl.convolution_param.weight_filler = ...
                                            PARAM.switchPredictionGstd{j};
                                    elseif strcmp(PARAM.switchPredictionGstd{j},'xavier')
                                        tsl.convolution_param.weight_filler = ...
                                            struct('type', {{'xavier'}});
                                    else
                                        error('Wrong switchPredictionGstd');
                                    end
                                    tsl.convolution_param.bias_filler = ...
                                        struct('type', {{'constant'}}, ...
                                        'value', {0});
                                    tsl.param(1).lr_mult    = 1;
                                    tsl.param(1).decay_mult = 1;
                                    tsl.param(2).lr_mult    = 2;
                                    tsl.param(2).decay_mult = 0;
                                    tsl = caffeproto_basic_convcombo(tsl);
                                    tsl.conv_combo_param.aux_layers = {'ReLU'};
                                    ts = cat_struct(2,ts,tsl);
                                    preTopName = tsl.top{1};
                                end
                                tsl = struct();
                                tsl.name = {[t.bottom{1} '/pool-pred']};
                                tsl.type = {'Reshape'};
                                tsl.bottom = {preTopName};
                                tsl.top    = {[t.bottom{1} '/pool-pred']};
                                tsl.reshape_param.axis = 1;
                                tsl.reshape_param.num_axes = 1;
                                tsl.reshape_param.shape.dim = [poolK,-1];
                                ts = cat_struct(2,ts,tsl);
                                preTopName = tsl.top{1};
                                if strcmp(unpoolingMethod,'pred:hard')
                                    tsl = struct();
                                    tsl.name = {[t.bottom{1} '/pool-pred/hard']};
                                    tsl.type = {'ArgMaxOne'};
                                    tsl.bottom = {preTopName};
                                    tsl.top    = {[t.bottom{1} '/pool-pred/hard']};
                                    ts = cat_struct(2,ts,tsl);
                                else
                                    tsl = struct();
                                    tsl.name = {[p.top{1} '/pool-pred/soft']};
                                    tsl.type = {'Softmax'};
                                    tsl.bottom = {preTopName};
                                    tsl.top    = {[t.bottom{1} '/pool-pred/soft']};
                                    ts = cat_struct(2,ts,tsl);
                                end
                                
                                % add training loss
                                if PARAM.trainSwitchLossWeight>0
                                    p.pooling_param.mask_index_type = pbEnum( 'LOCAL' ); % for soft case
                                    p = caffeproto_convcombo_set_aux_blobs( ...
                                        p, 'Pooling', [], {[p.top{1} '/pool-mask']} );
                                    tsl = struct();
                                    tsl.name = {[p.top{1} '/pool-gt']};
                                    tsl.type = {'Reshape'};
                                    tsl.bottom = {[p.top{1} '/pool-mask']};
                                    tsl.top    = {[p.top{1} '/pool-gt']};
                                    tsl.reshape_param.axis = 1;
                                    tsl.reshape_param.num_axes = 1;
                                    tsl.reshape_param.shape.dim = [1,-1];
                                    ts = cat_struct(2,ts,tsl);
                                    
                                    tsl = struct();
                                    tsl.name = {[PARAM.decoderPrefix 'switch-loss@' p.top{1}]};
                                    tsl.type = {'SoftmaxWithLoss'};
                                    tsl.bottom = {preTopName,[p.top{1} '/pool-gt']};
                                    tsl.top    = tsl.name;
                                    tsl.propagate_down = [true false];
                                    tsl.loss_weight = PARAM.trainSwitchLossWeight;
                                    ts = cat_struct(2,ts,tsl);
                                    
                                    tsl = struct();
                                    tsl.name = {[PARAM.decoderPrefix 'switch-accuracy@' p.top{1}]};
                                    tsl.type = {'Accuracy'};
                                    tsl.bottom = {preTopName,[p.top{1} '/pool-gt']};
                                    tsl.top    = tsl.name;
                                    tsl.propagate_down = [true false];
                                    tsl.include.phase = pbEnum('TEST');
                                    
                                    ts = cat_struct(2,ts,tsl);
                                                                        
                                end
                            otherwise
                                error( 'Unknown unpooling mask type (unpoolingMask)' );
                        end
                    end
                    if abs(PARAM.poolAveWeight)>eps
                        assert( isFida, 'poolAveWeight only works with Fida' )
                        if abs(PARAM.poolAveScale)<eps 
                            t.pooling_param.bias_weight  = PARAM.poolAveWeight;
                            t.fida_conv_param.pre_scaled = false;
                            if t.pooling_param.kernel_size>t.pooling_param.stride
                                t.pooling_param.array_normalized = [true];
                            end
                        else
                            t.pooling_param.pool(2) = pbEnum('AVE');
                            t.pooling_param.mult_scales  = [1,PARAM.poolAveScale];
                            t.pooling_param.mult_weights = [1,PARAM.poolAveWeight*t.pooling_param.kernel_size.^2];
                        end
                        if PARAM.poolAveCoeffLearnable
                            t.pooling_param.weight_learnable = true;
                            t.pooling_param.weight_lr_mult   = PARAM.poolAveWeightLRmult;
                        end
                    end
                
                    if isFida && PARAM.useFida>0
                        t = caffeproto_convcombo_set_aux_blobs( ...
                            t, 'Unpooling', [], {[t.bottom{1} '/unpool-mask']} );
                        t = caffeproto_convcombo_set_aux_blobs( ...
                            t, 'ConvCombo', {[t.bottom{1} '/unpool-mask']}, [] );
                        t.propagate_down = [true false]; % hard coded (Fida Layer doesn't backpropagate fidelity)
                    else
                        if PARAM.unpoolNormalization && t.pooling_param.kernel_size>t.pooling_param.stride
                            t.pooling_param.normalized = true;
                        end
                    end
                end
                if X.cae_stack_param.nonnegative_input
                    t.conv_combo_param.aux_layers = union( ...
                        t.conv_combo_param.aux_layers, {'ReLU'}, 'stable' );
                else
                    t.conv_combo_param.aux_layers = setdiff( ...
                        t.conv_combo_param.aux_layers, {'ReLU'}, 'stable' );
                end
                X.cae_stack_param.nonnegative_input = ismember( ...
                    {'ReLU'}, p.conv_combo_param.aux_layers );

                if has_pooling0 && PARAM.enlargeKernel
                    conv_kernel = t.convolution_param.kernel_size;
                    pool_stride = t.pooling_param.stride;
                    ks_margin = ceil((pool_stride-1)/2);
                    new_conv_kernel = conv_kernel+ks_margin*2;
                    if isfield( t.convolution_param, 'pad' )
                        t.convolution_param.pad = t.convolution_param.pad + ks_margin;
                    else
                        t.convolution_param.pad = ks_margin;
                    end
                    t.convolution_param.kernel_size = new_conv_kernel;
                end
                
                if ~isempty(l)
                    if strcmp( PARAM.linkTypeAtLoss, 'ladder' )
                        %assert(~isempty(strfind(l(end).type{1},'Loss')),'incompatible l');
                        lc = struct();
                        lc.name = { [ t.top{1} '/ladder-combined' ] };
                        lc.type = { 'LadderCombinator0' };
                        lc.top  = { [ t.top{1} '/ladder-combined' ] };
                        lc.bottom = { t.top{1}, p.bottom{1}}; % [top-down, skip-link]
                        % lc's top is not linked with any layer, which is ok as
                        % a intermediate step
                    end
                end

                addNoise4thisLayer = {};
                if islogical(PARAM.addNoise2Encoder) || isnumeric(PARAM.addNoise2Encoder)
                    assert( isscalar(PARAM.addNoise2Encoder), 'unrecognized addNoise2Encoder' );
                    if PARAM.addNoise2Encoder
                        if X.cae_stack_param.step==1
                            addNoise4thisLayer = [addNoise4thisLayer, {'before'}];
                        end
                        if ~isempty(X.layers)
                            addNoise4thisLayer = [addNoise4thisLayer, {'after'}];
                        end
                    end
                elseif iscell( PARAM.addNoise2Encoder )
                    if X.cae_stack_param.step==1
                        if ismember( p.bottom{1}, PARAM.addNoise2Encoder )
                            addNoise4thisLayer = [addNoise4thisLayer, {'before'}];
                        end
                    end
                    if ~isempty(X.layers)
                        if ismember( p.top{1}, PARAM.addNoise2Encoder )
                            addNoise4thisLayer = [addNoise4thisLayer, {'after'}];
                        end
                    end
                else
                    error( 'unrecognized addNoise2Encoder' );
                end
                if ismember( 'before', addNoise4thisLayer )
                    if ismember( 'AddNoise.Before', p.conv_combo_param.aux_layers )
                        warning( 'Already has noise aux_layer' );
                    end
                    p.conv_combo_param.aux_layers = [ p.conv_combo_param.aux_layers, {'AddNoise.Before'} ];
                    p.noise_param_before.type = { 'gaussian' };
                    p.noise_param_before.adaptive = boolean( PARAM.encoderNoiseAdaptive );
                end
                if ismember( 'after', addNoise4thisLayer )
                    if ismember( 'AddNoise.After', p.conv_combo_param.aux_layers )
                        warning( 'Already has noise aux_layer' );
                    end
                    if ismember('BN',p.conv_combo_param.aux_layers)
                        p.bn_param.noise_param.type = { 'gaussian' };
                        p.bn_param.noise_param.adaptive = false;
                    else
                        [has_relu, relu_idx ] = ismember('ReLU',p.conv_combo_param.aux_layers);
                        if has_relu
                            p.conv_combo_param.aux_layers = [ ...
                                p.conv_combo_param.aux_layers(1:relu_idx-1), {'AddNoise.After'}, ...
                                p.conv_combo_param.aux_layers(relu_idx:end) ];
                        else
                            p.conv_combo_param.aux_layers = [ p.conv_combo_param.aux_layers, {'AddNoise.After'} ];
                        end
                        p.noise_param_after.type = { 'gaussian' };
                        p.noise_param_after.adaptive = boolean( PARAM.encoderNoiseAdaptive );
                    end
                end
            else % for non-conv-combo
                t = p;
                t.name = {[PARAM.decoderPrefix p.name{1}]};
                t.bottom = {[PARAM.decoderPrefix p.top{:}]};
                t.top  = {[PARAM.decoderPrefix p.bottom{:}]};
                curInputShape = caffeproto_avail_ioshape( ...
                    p, 't', 1 );
                if ~isempty( curInputShape )
                    X.cae_stack_param.input_channels = curInputShape;
                end
            end
            %{
            if PARAM.encoderZeroLR
                if isfield(p,'param') && isfield(p.param,'lr_mult')
                    for k = 1:length(p.param)
                        if ~isempty( p.param(k).lr_mult )
                            p.param(k).lr_mult = 0;
                        end
                    end
                end
            end
            %}
            
            % first step
            if X.cae_stack_param.step==1
                % add recon loss at first step
                if PARAM.trainReconLossWeight>0
                    if strcmp(PARAM.trainWithReconRef,'@auto')
                        blobTrainRef = p.bottom{1};
                    else
                        blobTrainRef = PARAM.trainWithReconRef;
                    end
                    pl = struct([]);
                    if PARAM.trainRefScale ~= 1
                        pl = struct();
                        pl.name = {[blobTrainRef '/scaled']};
                        pl.type = {'Power'};
                        pl.bottom = {blobTrainRef};
                        if PARAM.enableNonprefixable, 
                            pl.nonprefixable_bottom = [1];
                        end
                        pl.top = pl.name;
                        pl.power_param.scale = PARAM.trainRefScale;
                        blobTrainRef = pl.top{1};
                    end
                    %
                    l = struct();
                    l.type = {'EuclideanLoss'};
                    l.name = {[PARAM.decoderPrefix 'loss@' blobTrainRef ]};
                    l.bottom = {t.top{1}, blobTrainRef};
                    if PARAM.enableNonprefixable, 
                        l.nonprefixable_bottom = 2;
                    end
                    l.top = l.name;
                    l.loss_weight = PARAM.trainReconLossWeight;
                    
                    dl = struct([]);
                    if PARAM.dumpImage
                        dl = repmat(struct(),1,2);
                        pm = struct([]);
                        am = struct([]);
                        blobDumpRef  = blobTrainRef;
                        blobDumpPred = l.bottom{1};
                        if ~isempty( PARAM.dumpMeanRef )
                            blobMeanRef = PARAM.dumpMeanRef;
                            if PARAM.trainRefScale ~= 1
                                pm = struct();
                                pm.name = {[blobMeanRef '/scaled']};
                                pm.type = {'Power'};
                                pm.bottom = {blobMeanRef};
                                if PARAM.enableNonprefixable, 
                                    pm.nonprefixable_bottom = 1;
                                end
                                pm.top = pm.name;
                                pm.power_param.scale = PARAM.trainRefScale;
                                blobMeanRef = pm.top{1};
                            end
                            am = repmat(struct(),1,2);
                            am(1).name   = {[blobDumpRef '/add-mean']};
                            am(1).type   = {'Eltwise'};
                            am(1).bottom = {blobDumpRef,blobMeanRef};
                            am(1).top    = am(1).name;
                            am(1).eltwise_param.operation = pbEnum('SUM');
                            blobDumpRef  = am(1).top{1};
                            am(2).name   = {[blobDumpPred '/add-mean']};
                            am(2).type   = {'Eltwise'};
                            am(2).bottom = {blobDumpPred,blobMeanRef};
                            am(2).top    = am(2).name;
                            am(2).eltwise_param.operation = pbEnum('SUM');
                            blobDumpPred = am(2).top{1};
                        end
                        dl(1).name = { [PARAM.decoderPrefix 'pair@' blobDumpRef] };
                        dl(1).type = { 'Concat' };
                        dl(1).bottom = {blobDumpRef,blobDumpPred};
                        dl(1).top  = dl(1).name;
                        dl(1).concat_param.axis = 3;
                        dl(2).name = { [PARAM.decoderPrefix 'dump@' blobDumpRef] };
                        dl(2).type = { 'ImageOutput' };
                        dl(2).bottom = dl(1).top;
                        dl(2).image_output_param.output_prefix=[ fullfile( 'dump', ...
                            strrep(strrep(blobDumpPred,':','_'),'/','_') ) '/'];
                        
                        dl = cat_struct(2, pm,am,dl);
                    end
                    
                    l=cat_struct(2,pl,l,dl);
                    X.cae_stack_param.added_loss = ...
                        [X.cae_stack_param.added_loss, p.bottom(1)];
                    
                end
            end

            is_loss_added = ismember( p.bottom(1), X.cae_stack_param.added_loss );
            is_at_head = isempty(X.layers) || strcmp( X.layers(1).bottom{1}, p.bottom{1} );
            if ~is_loss_added && ~is_at_head && isempty(l)
                if isnumeric( PARAM.trainInterReconLossWeight )
                    if X.cae_stack_param.step>1
                        thisInterReconLossWeight = PARAM.trainInterReconLossWeight;
                    else
                        thisInterReconLossWeight = -1;
                    end
                else
                    assert( iscell(PARAM.trainInterReconLossWeight) && ...
                        size(PARAM.trainInterReconLossWeight,2) == 2, ...
                        'trainInterReconLossWeight should either be a numeric scalar, or a N*2 cell array' );
                    [has_loss, loss_idx] = ismember( p.bottom{1} , PARAM.trainInterReconLossWeight(:,1) );
                    if has_loss,
                        thisInterReconLossWeight = PARAM.trainInterReconLossWeight{loss_idx,2};
                    else
                        thisInterReconLossWeight = -1;
                    end
                end
                assert( isscalar(thisInterReconLossWeight) && ...
                    isnumeric(thisInterReconLossWeight), ...
                    'loss weight must be numerc scalar' );
                if thisInterReconLossWeight>=0
                    
                    if strcmp( PARAM.linkTypeAtLoss, 'encoder' )
                        % this condition is guaranteed
                        % if ~isempty( X.layers ) % this condition avoids dup connect
                            splc = struct();
                            splc.name = { [X.name{1} '/connect@' p.bottom{1} ] };
                            splc.type = {'CAEStack'};
                            splc.top  = []; %t.top(1);
                            splc.bottom = p.bottom(1);
                            splc.layers = struct([]);
                        % end
                        % t.TOP = t.top; % TOP for pending revision
                        % t.TOP{1} = [t.top{1} '[2]'];
                        new_top_name = [t.top{1} '[LDB]']; % layerwise dec branch
                        if ~isempty(t.bottom)
                            revise_bottom_idxb = strcmp( t.bottom, t.top{1} );
                            if any(revise_bottom_idxb)
                                assert( sum(revise_bottom_idxb)==1, 'weird setting two inplace bottom' );
                                revise_bottom_idx = find(revise_bottom_idxb,1);
                                t.bottom{revise_bottom_idx} = new_top_name;
                                X.cae_stack_param.propagate_top = new_top_name;
                            end
                        end
                        t.top{1} = new_top_name;
                    end
                    
                    l = struct();
                    blobTrainRef = p.bottom{1};
                    l.type = {'EuclideanLoss'};
                    l.name = {[PARAM.decoderPrefix 'loss@' blobTrainRef ]};
                    l.bottom = {t.top{1}, blobTrainRef};
                    if PARAM.enableNonprefixable, 
                        l.nonprefixable_bottom = 2;
                    end
                    l.top = l.name;
                    l.loss_weight = thisInterReconLossWeight;
                    
                    X.cae_stack_param.added_loss = ...
                        [X.cae_stack_param.added_loss, p.bottom(1)];

                end
                
            end
            
            if ~isempty(X.cae_stack_param.propagate_top)
                is_dec_inplace = try_to_eval( 'strcmp(t.bottom{1},t.top{1})', false );
                t.top{1} = X.cae_stack_param.propagate_top;
                if is_dec_inplace
                    t.bottom{1} = X.cae_stack_param.propagate_top;
                else
                    X.cae_stack_param.propagate_top = [];
                end
            end
            
%             if X.cae_stack_param.step == 1 && ismember( p.type{1}, {'PReLU', 'ReLU'} )
%                 t = struct([]);
%             end
            
            X = cat_struct(2,p,X,ts,t,lc,l,splc);
            
        end
    otherwise
        error( 'unrecognized branch' );
end

