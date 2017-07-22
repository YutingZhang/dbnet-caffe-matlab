function X = caffeproto_replace_func_add_propagated_tile( varargin )

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

is_matched = default_eval( 'subS.num_output_factor', 1 ) < 1 && ...
    strcmp( subS.type{1}, 'ConvCombo' ) && ...
    ~ismember( 'ReLU', subS.conv_combo_param.aux_layers );

if ~is_matched, return; end

X = subS;
X.conv_combo_param.aux_layers = [ ...
    subS.conv_combo_param.aux_layers, {'ReLU'} ];

X.relu_param.crelu = 1;
X.relu_param.crelu_tile_only = 1;
