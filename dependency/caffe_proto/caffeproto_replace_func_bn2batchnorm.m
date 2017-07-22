function X = caffeproto_replace_func_bn2batchnorm( varargin )

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

isValid = strcmp( subS.type{1}, 'BN' );
if ~isValid, return; end

bn_mode = try_to_eval( 'subS.bn_param.bn_mode.val', 'LEARN' );

is_correction = ismember( bn_mode, {'CORRECTION','AUTO_CORRECT'} );

% use_global_stats = try_to_eval( 'subS.bn_param.batch_norm_param.use_global_stats', [] );
bn_param = default_eval( 'subS.bn_param', struct() );
batch_norm_param = try_to_eval( 'bn_param.batch_norm_param', struct() );
batch_norm_param = partial_struct( batch_norm_param, '@exclude', 'use_global_stats' );

X = partial_struct( subS, 'name', 'type', 'bottom', 'top' );
if is_correction
    X.type = {'Scale'};
else
    X.type = {'BatchNorm'};
end

if ~is_correction
    X.batch_norm_param = batch_norm_param;

    switch bn_mode
        case 'LEARN'
            X.batch_norm_param.has_scale_layer = true;
        case 'CORRECT'
            X.batch_norm_param.has_scale_layer  = true;
            X.batch_norm_param.use_global_stats = true;
        case 'AUTO_NORM'
            X.batch_norm_param.has_scale_layer  = false;
            % X.batch_norm_param.use_global_stats = [];
        case 'INFERENCE'
            X.batch_norm_param.has_scale_layer  = false;
            X.batch_norm_param.use_global_stats = true;
        case 'NORM'
            X.batch_norm_param.has_scale_layer  = false;
            X.batch_norm_param.use_global_stats = false;
        otherwise
            X = [];
            return;
    end
end

if X.batch_norm_param.has_scale_layer
    % param for scale layer
    X = xmerge_struct( 'always', 'always', X, partial_struct( subS, 'param' ) );
else
    % param for itself
    X.param = struct( 'lr_mult', {0,0,0}, 'decay_mult', {0, 0, 0} );
end

scale_param = [];
if is_correction
    scale_param = try_to_eval( 'X.scale_param', struct() );
    scale_param.bias_term = true; % use shift by default
elseif X.batch_norm_param.has_scale_layer
    if isfield( X.batch_norm_param, 'scale_param' )
        scale_param = X.batch_norm_param.scale_param;
    else
        scale_param = struct();
        scale_param.bias_term = true; % use shift by default
    end
end

if ~isempty(scale_param)
    [scale_param, ~] = transfer_field( scale_param, bn_param, {'filler', 'scale_filler'}, 1 );
    [scale_param, ~] = transfer_field( scale_param, bn_param, {'bias_filler', 'shift_filler'}, 1 );
    [scale_param, ~] = transfer_field( scale_param, bn_param, 'axis', 1 );
    scale_param = rm_empty_field( scale_param );
    if isempty( fieldnames(scale_param) ), scale_param = [] ; end
    if is_correction
        X.scale_param = scale_param;
    else
        X.batch_norm_param.scale_param = scale_param;
    end
end

