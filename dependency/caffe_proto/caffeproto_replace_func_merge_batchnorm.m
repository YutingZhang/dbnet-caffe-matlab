function [X, varargout] = caffeproto_replace_func_merge_batchnorm( varargin )
% 

if strcmp( varargin{1}, 'extend' )
    X = {};
    return;
end

if strcmp( varargin{1}, 'adjacent' ) 
    X = [
        0 1 
        0 0 ];
    varargout = { [ 1 ], [ 2 ] };
    return;
end

X = [];

subS = varargin{1};

isValid = strcmp( subS(1).type, 'BatchNorm' ) && strcmp( subS(2).type, 'Scale' );

if ~isValid, return; end

X = partial_struct( subS(1), '@exclude', 'param' );
X.name = subS(2).name;
X.top  = subS(2).top;
X.batch_norm_param.has_scale_layer = true;
X.batch_norm_param.scale_param = try_to_eval( 'subS(2).scale_param', [] );
X = xmerge_struct( 'always', 'always', X, partial_struct( subS(2), 'param' ) );

