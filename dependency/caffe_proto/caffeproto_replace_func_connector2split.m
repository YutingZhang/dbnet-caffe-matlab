function X = caffeproto_replace_func_connector2split( varargin )

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

isValid = strcmp( subS.type{1}, 'Reshape' );
if ~isValid, return; end

isValid = try_to_eval( 'subS.reshape_param.num_axes', -1 ) == 0 && ...
    isempty( try_to_eval( 'subS.reshape_param.shape.dim', [] ) );
if ~isValid, return; end

X = partial_struct( subS, 'name', 'bottom', 'top' );
X.type = { 'Split' };

