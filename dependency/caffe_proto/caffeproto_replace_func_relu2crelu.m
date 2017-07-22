function X = caffeproto_replace_func_relu2crelu( varargin )

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

is_matched = strcmp( subS.type{1}, 'ReLU' ) || ...
    ( strcmp( subS.type{1}, 'ConvCombo' ) && ...
    ismember( 'ReLU', subS.conv_combo_param.aux_layers ) );
if ~is_matched, return; end

ARGS = varargin{end};
if numel(ARGS)>=1
    half_paramters = ARGS{1};
else
    half_paramters = false;
end

is_crelu = default_eval( 'subS.relu_param.crelu', false );
if ~is_crelu
    ConvTypes = {'Convolution', 'Deconvolution', 'FidaConv', 'FidaDeconv'};
    X = subS;
    if strcmp( subS.type{1}, 'ReLU' ) || ismember( subS.conv_combo_param.type, ConvTypes )
        X.relu_param.crelu = 1;
    end
    if half_paramters
        X.num_output_factor = 0.5 * default_eval( 'subS.num_output_factor', 1 );
    end
end

