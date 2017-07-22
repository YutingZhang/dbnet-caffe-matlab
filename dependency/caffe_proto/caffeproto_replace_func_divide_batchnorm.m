function X = caffeproto_replace_func_divide_batchnorm( varargin )
% 

if strcmp( varargin{1}, 'extend' )
    X = {};
    return;
end

if strcmp( varargin{1}, 'adjacent' )
    X = [0];
    return;
end


subS = varargin{1};

has_scale_layer = try_to_eval( 'subS.batch_norm_param.has_scale_layer', [] );
isValid = strcmp( subS.type{1}, 'BatchNorm' ) && ~isempty( has_scale_layer );

X = [];

if ~isValid, return; end

if ~has_scale_layer
    X = subS;
    X.batch_norm_param = rmfield( X.batch_norm_param, 'has_scale_layer' );
    return;
end

% use general param first
%PARAM = scalar_struct(varargin{end}{:});
%PARAM = xmerge_struct('always','always', Pdef, PARAM);

X = struct([]);

G = partial_struct( subS, '@exclude', 'param' );
G.name = { [subS.name{1} '/norm'] };
G.batch_norm_param = partial_struct( subS.batch_norm_param, '@exclude', ...
    'has_scale_layer', 'scale_param' );
G.param = struct( 'lr_mult', {0,0,0}, 'decay_mult', {0, 0, 0} );
X = cat_struct( 2, X, G );

G = struct();
G.name = { [subS.name{1}] };
G.type = {'Scale'};
G.bottom = X(1).top;
G.top    = G.bottom;
G.scale_param = subS.batch_norm_param.scale_param;
G = xmerge_struct( 'always', 'always', G, partial_struct( subS, 'param' ) );

X = cat_struct( 2, X, G );

