function X = caffeproto_replace_func_recognize_global_pooling( varargin )
% 

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
if ~strcmp( subS.type, 'Pooling' ), return; end

bottomShape = caffeproto_avail_ioshape( subS, 'b', 1 );
if length(bottomShape)<=1, return; end

kernelSize = default_eval( 'subS.pooling_param.kernel_size', [] );

if isempty(kernelSize), return; end

geoShape = bottomShape(2:end);

if isscalar( kernelSize )
    kernelSize = repmat( kernelSize, size(geoShape) );
end

is_matched = isequal( kernelSize, geoShape );

if ~is_matched, return; end

X = subS;
X.pooling_param  = rmfield( X.pooling_param, 'kernel_size' );
X.pooling_param.global_pooling = true;

