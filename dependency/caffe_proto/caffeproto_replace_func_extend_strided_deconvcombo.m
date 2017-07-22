function [X, varargout] = caffeproto_replace_func_extend_strided_deconvcombo( varargin )

if strcmp( varargin{1}, 'extend' )
    X = [];
    return;
elseif strcmp( varargin{1}, 'adjacent' )
    X = [0];
    return;
end

subS = varargin{1};

DeconvTypes = { 'Deconvolution', 'FidaDeconv' };
ConvTypes   = { 'Convolution',   'FidaConv' };

X = [];

is_matched = strcmp( subS.type, 'ConvCombo' ) && ...
    ~subS.conv_combo_param.is_strided_leaf && ...
    ismember( subS.conv_combo_param.type, DeconvTypes );
if ~is_matched, return; end

is_matched = isfield( subS, 'convolution_param' ) && ...
    isfield( subS.convolution_param, 'stride' ) && ...
    subS.convolution_param.stride>1;
if ~is_matched, return; end

ARGS = varargin{end};

if isfield( subS.conv_combo_param, 'input_channels' )
    inCh = subS.conv_combo_param.input_channels;
else
    assert( ~isempty(ARGS) && ~isempty(ARGS{1}), ...
        'need to know the input channel number' );
    inCh = ARGS{1};   % ARGS{1}
end

[~,typeIdx]=ismember( subS.conv_combo_param.type, DeconvTypes );
thisConvType = ConvTypes(typeIdx);

X = struct([]);
subS.conv_combo_param.is_strided_leaf = 1;
if ismember( 'Unpooling', subS.conv_combo_param.aux_layers )
    G = subS;
    G.conv_combo_param.aux_layers = {'ReLU','Unpooling'};
    
    ck = floor((G.pooling_param.stride+G.pooling_param.kernel_size)/2)*2+1;
    
    G.convolution_param.stride = 1;
    G.convolution_param.pad    = floor(ck/2);
    G.convolution_param.kernel_size = ck;
    G.convolution_param.num_output  = inCh(1);
    
    G.name = { [ subS.name{1} '-pre' ] };
    % use conv instead of deconv for legacy reason
    G.conv_combo_param.type = thisConvType;
    G.top = {[subS.bottom{1} '-pre']};
    
    X = cat_struct(2, X, G );
    subS.conv_combo_param.aux_layers = setdiff( ...
        subS.conv_combo_param.aux_layers, {'Unpooling'}, 'stable' );
    subS = caffeproto_convcombo_set_aux_blobs( subS, 'ConvCombo', {}, {} );    
    subS = caffeproto_convcombo_set_aux_blobs( subS, 'Unpooling', {}, {} );    
    subS = partial_struct( subS, '@exclude', 'pooling_param', 'propagate_down' );
end

G = subS;
G.conv_combo_param.aux_layers = union( ...
    {'ReLU'}, subS.conv_combo_param.aux_layers, 'stable' );
G.convolution_param.num_output  = max(ceil(inCh(1)/2), G.convolution_param.num_output*2);
G.name = { [ subS.name{1} '-body' ] };
if ~isempty(X)
    G.bottom = X(end).top;
end
G.top = { [subS.bottom{1} '-body'] };
X = cat_struct(2, X, G);

G = subS;
conv_combo_param.aux_layers = intersect( ...
    subS.conv_combo_param.aux_layers, {'ReLU'}, 'stable' );
ck = floor(subS.convolution_param.stride/2)*2+1;
G.convolution_param.stride = 1;
G.convolution_param.pad    = floor(ck/2);
G.convolution_param.kernel_size = ck;
G.bottom = X(end).top;
G.name = { [ subS.name{1} '-post' ] };
X = cat_struct(2, X, G);

