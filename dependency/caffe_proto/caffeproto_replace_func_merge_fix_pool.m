function X = caffeproto_replace_func_merge_fix_pool( varargin )

%
if strcmp( varargin{1}, 'extend' )
    X = { };
    return;
end

if strcmp( varargin{1}, 'adjacent' )
    X = [0];
    return;
end

X = [];

subS = varargin{1};

is_matched = strcmp( subS.type{1}, 'ConvCombo' ) && ...
    ismember( subS.conv_combo_param.type{1}, {'Convolution','FidaConv'} ) && ...
    ismember( 'Pooling', subS.conv_combo_param.aux_layers );

if ~is_matched, return; end

is_matched = strcmp( subS.pooling_param.pool.val, 'FIX' );

if ~is_matched, return; end

pool_stride  = default_eval( 'subS.pooling_param.stride', 1 );
conv_stride0 = default_eval( 'subS.convolution_param.stride', 1 );
conv_stride1 = conv_stride0.*pool_stride;

X = rmfield(subS,'pooling_param');
X.conv_combo_param.aux_layers = setdiff( X.conv_combo_param.aux_layers, ...
    {'Pooling'} );
X.convolution_param.stride = conv_stride1;

