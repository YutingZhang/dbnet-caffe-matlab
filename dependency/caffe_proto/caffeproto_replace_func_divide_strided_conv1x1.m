function X = caffeproto_replace_func_divide_strided_conv1x1( varargin )

if strcmp(varargin{1},'extend')
    X = {};
    return;
elseif strcmp(varargin{1},'adjacent')
    X = [0];
    return;
end

subS = varargin{1};

X = [];

if ~ismember( subS.type{1}, {'Convolution'} )
    return;
end

s = default_eval( 'subS.convolution_param.stride', 1 );
if all(s(1)==s), s=s(1); end
is_matched = any( s > 1 ) && ...
    all( default_eval( 'subS.convolution_param.kernel_size', 1 ) == 1 );

if ~is_matched, return; end

p = default_eval( 'subS.convolution_param.pad', 0 );
assert( isempty(p) || ~any(p), 'Do not support padding. Not implemented' );


ARGS = varargin{end};
poolingType = default_eval( 'ARGS{1}', 'fix' );


X = subS;
X.convolution_param.kernel_size = 1;
X.convolution_param.stride      = 1;

G = struct();
G.name = {[subS.name{1} '/pre-pool']};
G.type = {'Pooling'};
G.bottom = subS.bottom(1);
G.top    = G.name;
if length(s)>1
    G.pooling_param.kernel_h = s(1);
    G.pooling_param.kernel_w = s(2);
    G.pooling_param.stride_h = s(1);
    G.pooling_param.stride_w = s(2);
else
    G.pooling_param.kernel_size = s;
    G.pooling_param.stride = s;
end

X.bottom(1) = G.top(1);

if strcmpi( poolingType, 'fix' )
    G.pooling_param.pool = pbEnum('FIX');
    G.pooling_param.fix = default_eval( 'ARGS{2}', 0 );
elseif strcmpi( poolingType, 'fix:center' )
    G.pooling_param.pool = pbEnum('FIX');
    G.pooling_param.fix = -1;
else
    G.pooling_param.pool = pbEnum(upper(poolingType));
end

X = cat_struct(2,G,X);

