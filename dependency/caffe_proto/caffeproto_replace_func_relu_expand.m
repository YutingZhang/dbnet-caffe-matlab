function X = caffeproto_replace_func_relu_expand( varargin )

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

if ~strcmp( subS.type{1}, 'ReLU' ), return; end

nonlinearityType = default_eval( 'subS.relu_param.type{1}', 'ReLU' );

isCReLU = default_eval( 'subS.relu_param.crelu', false );

if strcmp( nonlinearityType, 'ReLU' ) && ~isCReLU , return, end

X = rmfield( subS, 'relu_param' );
switch nonlinearityType
    case {'LReLU','PReLU'}
        X.type = { 'PReLU' };
        X.prelu_param = struct();
        X.prelu_param.filler = struct( ...
            'type', {'constant'}, 'value', ...
            try_to_eval( 'subS.relu_param.leakage', 0.15 ) );
        if strcmp(nonlinearityType,'LReLU')
            X.param = struct( 'lr_mult', 0, 'decay_mult', 0 );
            X.prelu_param.channel_shared = true;
        else % PReLU
            X.param = struct( 'lr_mult', 1, 'decay_mult', 0 );
        end
    otherwise
        X.type = { nonlinearityType };
end

if isCReLU
    G = struct();
    G.name = { [ X.name{1} '/dup-flip' ] };
    G.type = { 'Tile' };
    G.bottom = X.bottom(1);
    G.top    = X.top(1);
    G.tile_param.tiles = 2;
    G.tile_param.coeff = [1 -1];
    X.bottom = X.top(1);
    if default_eval( 'subS.relu_param.crelu_tile_only', false )
        X = G;
    else
        X = cat_struct(2, G, X);
    end
end

