function X = caffeproto_replace_func_batchnorm2bn( varargin )


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

if default_eval( 'subS.batch_norm_param.has_scale_layer', false )
    % record scale param
    X = partial_struct( subS, 'name', 'type', 'bottom', 'top', 'param' );
else
    X = partial_struct( subS, 'name', 'type', 'bottom', 'top' );
end
X.type = {'BN'};

use_global_stats = try_to_eval( 'subS.batch_norm_param.use_global_stats', [] );
batch_norm_param = try_to_eval( 'subS.batch_norm_param', struct() );
batch_norm_param = partial_struct( batch_norm_param, '@exclude', ...
    'has_scale_layer', 'scale_param', 'use_global_stats' );

if try_to_eval( 'subS.batch_norm_param.has_scale_layer', false )
    % Norm + Scaling
    if isempty(use_global_stats)
        X.bn_param.bn_mode = pbEnum( 'LEARN' );
    elseif use_global_stats
        X.bn_param.bn_mode = pbEnum( 'CORRECT' );
    else
        X.bn_param.bn_mode = pbEnum( 'LEARN' );
        batch_norm_param.use_global_stats = false;
    end
    scale_param = try_to_eval( ...
        'subS.batch_norm_param.scale_param', false );
else
    % Norm
    if isempty(use_global_stats)
        X.bn_param.bn_mode = pbEnum( 'AUTO_NORM' );
    elseif use_global_state
        X.bn_param.bn_mode = pbEnum( 'INFERENCE' );
    else
        X.bn_param.bn_mode = pbEnum( 'NORM' );
    end
    scale_param = [];
end

if ~isempty( fieldnames( batch_norm_param ) )
    X.bn_param.batch_norm_param = batch_norm_param;
end

if ~isempty(scale_param)
    bn_param = X.bn_param;
    [bn_param, scale_param] = transfer_field( bn_param, scale_param, {'scale_filler', 'filler'}, 1 );
    [bn_param, scale_param] = transfer_field( bn_param, scale_param, {'shift_filler', 'bias_filler'}, 1 );
    [bn_param, scale_param] = transfer_field( bn_param, scale_param, 'axis', 1 );
    bn_param.batch_norm_param.scale_param = scale_param;
    X.bn_param = bn_param;
end


