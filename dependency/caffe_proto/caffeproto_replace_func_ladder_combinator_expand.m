function X = caffeproto_replace_func_ladder_combinator_expand( varargin )

if strcmp( varargin{1}, 'extend' )
    X = {};
    return;
end

if strcmp( varargin{1}, 'adjacent' )
    X = [0];
    return;
end

subS = varargin{1};

X = [];

if ~strcmp( subS.type{1}, 'LadderCombinator' )
    return;
end

Pdef = struct();
Pdef.type = 'AMLP';
Pdef.mlp_num_nodes    = [2 2];
Pdef.mlp_nonlinearity = 'LReLU'; % can be "ReLU", "LReLU", "PReLU"
Pdef.mlp_relu_leakage = 0.1;

% use general param first
PARAM = scalar_struct(varargin{end}{:});
PARAM = xmerge_struct('always','always', Pdef, PARAM);

% if there is layer spec, then use it instead
if isfield( subS, 'ladder_combined_param' ) && ~isempty(ladder_combined_param)
    PARAM = xmerge_struct('always',@(a) ~isempty(a), ...
        PARAM, subS.ladder_combined_param );
    if iscell( PARAM.type ), PARAM.type = PARAM.type{:}; end 
    if iscell( PARAM.mlp_nonlinearity ), PARAM.type = PARAM.mlp_nonlinearity{:}; end 
end

X = struct([]);
switch PARAM.type
    case {'MLP','AMLP'}
        % reshape to make it single channel
        
        R = struct([]);
        
        R(1).name = { [subS.name{1} '/reshape-bottom-1' ] };
        R(1).type = { 'Reshape' };
        R(1).bottom = subS.bottom(1);
        R(1).top    = R(1).name;

        R(1).reshape_param = struct();
        R(1).reshape_param = prototxt2struct( ...
            'shape { dim: 1 }  axis: 1  num_axes: 0', 'string' );
        
        R(2).name = { [subS.name{1} '/reshape-bottom-2' ] };
        R(2).type = { 'Reshape' };
        R(2).bottom = subS.bottom(2);
        R(2).top  = R(2).name;
        R(2).reshape_param = R(1).reshape_param;
        
        if strcmp(PARAM.type,'AMLP')
            % do multiplication
            R(3).name = { [subS.name{1} '/bottom-prod' ] };
            R(3).type = { 'Eltwise' };
            R(3).bottom = [R(1).top, R(2).top];
            R(3).top  = R(3).name;
            R(3).eltwise_param = struct();
            R(3).eltwise_param.operation = pbEnum('PROD');
        end
        
        % combine input
        C = struct();
        C.name = { [subS.name{1} '/concat' ] };
        C.type = { 'Concat' };
        C.bottom = [ R.top ];
        C.top  = C.name;
        C.concat_param.axis = 1; % channel
        
        X = cat_struct( 2, X, R, C );
        % generate MLP
        mlp_nodes = [PARAM.mlp_num_nodes,1];
        for k = 1:length(mlp_nodes)
            G = struct();
            G.name = { [subS.name{1} '/mlp' int2str(k) ] };
            G.type = { 'Convolution' };
            G.bottom = X(end).top(1);
            G.top  = G.name;
            G.convolution_param = struct();
            G.convolution_param.num_output  = mlp_nodes(k);
            G.convolution_param.kernel_size = 1;
            G.convolution_param.weight_filler = ...
                struct( 'type', { {'gaussian'} }, 'std', {0.001} );
            G.convolution_param.bias_filler = ...
                struct( 'type', { {'constant'} }, 'value', {0} );
            G.param = struct( 'lr_mult', {1,1}, 'decay_mult', {1,0} );
            G = caffeproto_basic_convcombo(G);
            if k<length(mlp_nodes)
                G.conv_combo_param.aux_layers{end+1} = 'ReLU';
                G.relu_param.type = { PARAM.mlp_nonlinearity };
                if ismember( PARAM.mlp_nonlinearity, {'LReLU','PReLU'} )
                    G.relu_param.leakage = PARAM.mlp_relu_leakage;
                end
            end
            X = cat_struct( 2, X, G );
        end
        
        % remove eltwise channel
        G = struct();
        G.name = { [subS.name{1} '/reshape-top' ] };
        G.type = { 'Reshape' };
        G.bottom = X(end).top(1);
        G.top    = subS.top(1);
        G.reshape_param = prototxt2struct( ...
            'shape { dim: -1 }  axis: 0  num_axes: 2', 'string' );

        X = cat_struct( 2, X, G );
        
    otherwise
        error( 'unrecognized combinator type' );
end
        