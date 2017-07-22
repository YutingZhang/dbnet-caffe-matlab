function X = caffeproto_replace_func_merge_bn( varargin )
% 

if strcmp( varargin{1}, 'extend' )
    X = {};
    return;
end


if strcmp( varargin{1}, 'adjacent' ) 
    X = [
        0 1 1
        0 0 1
        0 0 0];
    return;
end

X = [];

subS = varargin{1};

isValid = strcmp( subS(1).type, 'BN' ) && length( subS(1).top ) == 3 && ...
    strcmp( subS(2).type, 'RunningAverage' ) && ...
    length( subS(2).bottom ) == 2 && length( subS(2).top ) == 2 && ...
    strcmp( subS(3).type, 'BN' ) && length( subS(3).bottom ) == 3;

if ~isValid, return; end

X = struct();
X.name = subS(3).name;
X.type = {'BN'};
X.bottom = subS(1).bottom;
X.top  = subS(3).top;
X.param = try_to_eval( 'subS(3).param', [] );
X.bn_param = try_to_eval( 'subS(3).bn_param', [] );
X.bn_param.bn_mode = pbEnum( 'LEARN' );
