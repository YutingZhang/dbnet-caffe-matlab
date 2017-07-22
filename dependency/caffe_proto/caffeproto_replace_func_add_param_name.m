function X = caffeproto_replace_func_add_param_name( varargin )

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
isValid = isfield( subS, 'param' ) && ~isempty(subS.param);
if ~isValid
    return;
end

X = subS;
for k = 1:numel(X.param)
    if ~isfield( X.param(k), 'name' ) || isempty(X.param(k).name)
        X.param(k).name = { [ X.name{1} ':blob' int2str(k) ] };
    end
end

