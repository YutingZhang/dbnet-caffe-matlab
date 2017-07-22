function X = caffeproto_replace_func_expand_strided_conv2pool( varargin )

if strcmp(varargin{1},'extend')
    X = {};
    return;
elseif strcmp(varargin{1},'adjacent')
    X = [0];
    return;
end

subS = varargin{1};

X = [];

if ~ismember( subS.type{1}, {'ConvCombo'} ) ||  ...
        ~strcmp( subS.conv_combo_param.type{1}, 'Convolution' )
    return;
end

s = default_eval( 'subS.convolution_param.stride', 1 );
if all(s(1)==s), s=s(1); end
is_matched = any( s > 1 ) && ~ismember( 'Pooling', subS.conv_combo_param.aux_layers );

if ~is_matched, return; end

%p = default_eval( 'subS.convolution_param.pad', 0 );
%assert( isempty(p) || ~any(p), 'Do not support padding. Not implemented' );

ARGS = varargin{end};
poolingType = default_eval( 'ARGS{1}', 'fix' );

X = subS;
X.convolution_param.stride = 1;

X.conv_combo_param.aux_layers{end+1} = 'Pooling';

X.pooling_param.pool = pbEnum(upper(poolingType));
if length(s)>1
    X.pooling_param.kernel_h = s(1);
    X.pooling_param.kernel_w = s(2);
    X.pooling_param.stride_h = s(1);
    X.pooling_param.stride_w = s(2);
else
    X.pooling_param.kernel_size = s;
    X.pooling_param.stride = s;
end

