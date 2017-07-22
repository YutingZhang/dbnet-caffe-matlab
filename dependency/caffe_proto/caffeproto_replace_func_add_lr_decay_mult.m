function [X,varargout] = caffeproto_replace_func_add_lr_decay_mult( varargin )

if strcmp(varargin{1},'extend')
    X = {};
    return;
elseif strcmp(varargin{1},'adjacent')
    X = [0];
    return;
end

subS = varargin{1};

P0.force = 0;
P0.w_lr_mult = 1;
P0.w_decay_mult = 1;
P0.b_lr_mult = 1;
P0.b_decay_mult = 0;

ARGS = varargin{end};

ARGS(2:2:end) = cellfun( @(a) {a},ARGS(2:2:end), 'UniformOutput', 0 );
PARAM = struct( ARGS{:} );
PARAM = xmerge_struct('always','always',P0,PARAM);

WBMap = {
    'Convolution',  'convolution_param', true
    'Deconvoltion', 'convolution_param', true
    'FidaConv',     'convolution_param', true
    'FidaDeconv',   'convolution_param', true
    'InnerProduct', 'inner_product_param', true
    'Scale',        'scale_param', false };

X = [];

[is_wb, wb_loc] = ismember( subS.type{1}, WBMap(:,1) );
if is_wb
    X = subS;
    has_bias = default_eval( sprintf('subS.%s.bias_term', WBMap{wb_loc,2} ), WBMap{wb_loc,3} );
    X = set_param_for_X( X, PARAM, has_bias );
elseif strcmp( subS.type{1}, 'BatchNorm' )
    X = subS;
    if try_to_eval( 'subS.batch_norm_param.has_scale_layer', false )
        X.param = struct( 'lr_mult', {0,0,0}, 'decay_mult', {0,0,0} );
    else
        has_bias = default_eval( 'subS.batch_norm_param.scale_param.bias_term', false );
        X = set_param_for_X( X, PARAM, has_bias );
    end
end

function X = set_param_for_X( X, PARAM, has_bias )

if PARAM.force
    X.param(1).lr_mult    = PARAM.w_lr_mult;
    X.param(1).decay_mult = PARAM.w_decay_mult;
    if has_bias
        X.param(2).lr_mult    = PARM.b_lr_mult;
        X.param(2).decay_mult = PARM.b_decay_mult;
    end
else
    X.param(1).lr_mult    = default_eval( 'X.param(1).lr_mult',    PARAM.w_lr_mult );
    X.param(1).decay_mult = default_eval( 'X.param(1).decay_mult', PARAM.w_decay_mult );
    if has_bias
        X.param(2).lr_mult    = default_eval( 'X.param(2).lr_mult',    PARAM.b_lr_mult );
        X.param(2).decay_mult = default_eval( 'X.param(2).decay_mult', PARAM.b_decay_mult );
    end
end
