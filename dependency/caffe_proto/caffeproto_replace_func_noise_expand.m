function X = caffeproto_replace_func_noise_expand( varargin )

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
if ~strcmp( subS.type{1}, 'AddNoise' )
    return;
end

Pdef.enableNonprefixable = 0;

PARAM = scalar_struct(varargin{end}{:});
PARAM = xmerge_struct('always','always', Pdef, PARAM);


switch subS.noise_param.type{1}
    case 'gaussian'
        X = struct();
        X.name = subS.name;
        X.type = { 'EltwiseWithFiller' };
        X.bottom = subS.bottom;
        X.top    = subS.top;
        X.eltwise_with_filler_param = struct();
        X.eltwise_with_filler_param.filler = struct( ...
            'type', {'gaussian'}, 'std', {1} );
        X.eltwise_with_filler_param.eltwise = struct( ...
            'operation', { pbEnum('SUM') } , 'coeff', { [1 1] } );
    otherwise
        error( 'unrecognized noise_param.type' );
end

if try_to_eval( 'subS.noise_param.adaptive', false )
    G = struct();
    G.name = { [subS.bottom{1} '/bn'] };
    G.type = { 'BN' };
    G.bottom = X.bottom(1);
    G.top  = { G.name{1}, [subS.bottom{1} '/batch-mean'], [subS.bottom{1} '/batch-var'] };
    G.bn_param.bn_mode = pbEnum( 'NORM' );
    if PARAM.enableNonprefixable , G.nonprefixable_bottom = [1]; end
    
    T = struct();
    T.name = { [G.name{1} '/silence'] };
    T.type = { 'Silence' };
    T.bottom = G.top(1:2);
    
    X.bottom = [X.bottom, G.top(3)];
    X = cat_struct( 2, G, T, X );
end
