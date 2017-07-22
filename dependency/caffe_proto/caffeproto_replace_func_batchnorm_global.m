function X = caffeproto_replace_func_batchnorm_global( varargin )

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

isValid = strcmp( subS.type{1}, 'BatchNorm' );
if ~isValid, return; end

ARGS = varargin{end};
if isempty( ARGS )
    use_global_stats = [];
else
    use_global_stats = ARGS{1};
end

X = subS();
X.batch_norm_param.use_global_stats = use_global_stats;

if isempty(use_global_stats)
    X.batch_norm_param = rmfield( X.batch_norm_param, 'use_global_stats' );
end

