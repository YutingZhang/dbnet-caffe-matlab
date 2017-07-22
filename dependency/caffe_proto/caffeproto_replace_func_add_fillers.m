function [X,varargout] = caffeproto_replace_func_add_fillers( varargin )

if strcmp(varargin{1},'extend')
    X = {};
    return;
elseif strcmp(varargin{1},'adjacent')
    X = [0];
    return;
end

subS = varargin{1};

P0.force = 0;
P0.std   = 1e-3;
P0.value = 0;

ARGS = varargin{end};

ARGS(2:2:end) = cellfun( @(a) {a},ARGS(2:2:end), 'UniformOutput', 0 );
PARAM = struct( ARGS{:} );
PARAM = xmerge_struct('always','always',P0,PARAM);

ConvTypes = {'Convolution','Deconvoltion','FidaConv','FidaDeconv'};


X = subS;

if ismember( subS.type{1}, ConvTypes )
    if PARAM.force || ~isfield(X, 'convolution_param') || ... 
        ~isfield(X.convolution_param, 'weight_filler')
        X.convolution_param.weight_filler.type = 'gaussian';
        X.convolution_param.weight_filler.std  = PARAM.std;
    end
    if PARAM.force || ~isfield(X, 'convolution_param') || ... 
        ~isfield(X.convolution_param, 'bias_filler')
        X.convolution_param.bias_filler.type  = 'constant';
        X.convolution_param.bias_filler.value = PARAM.value;
    end
elseif strcmp( subS.type{1}, 'InnerProduct' )
    if PARAM.force || ~isfield(X, 'inner_product_param') || ... 
        ~isfield(X.inner_product_param, 'weight_filler')
        X.inner_product_param.weight_filler.type = 'gaussian';
        X.inner_product_param.weight_filler.std  = PARAM.std;
    end
    if ~isfield(X, 'inner_product_param') || ... 
        ~isfield(X.inner_product_param, 'bias_filler')
        X.inner_product_param.bias_filler.type  = 'constant';
        X.inner_product_param.bias_filler.value = PARAM.value;
    end
elseif strcmp( subS.type{1}, 'Scale' )
    if PARAM.force || isempty( try_to_eval( 'X.scale_param.filler.type{1}', [] ) )
        X.scale_param.filler.type  = { 'constant' };
        X.scale_param.filler.value = 1;
    end
    if PARAM.force || isempty( try_to_eval( 'X.scale_param.bias_filler.type{1}', [] ) )
        X.scale_param.bias_filler.type  = { 'constant' };
        X.scale_param.bias_filler.value = 0;
    end
else
    X = [];
end
