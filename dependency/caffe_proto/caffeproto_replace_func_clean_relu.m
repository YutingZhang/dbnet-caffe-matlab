function [X, varargout] = caffeproto_replace_func_clean_relu( varargin )

if strcmp( varargin{1}, 'extend' )
    X = { { 'list', 'iterative', ...
        @(varargin) caffeproto_replace_func_clean_relu( 'internal', varargin{:} ) } };
    return;
end

VAR_IN = varargin(2:end);

if strcmp(VAR_IN{1},'extend')
    X = {};
    return;
end

if strcmp( VAR_IN{1}, 'adjacent' )
    X = [0 1; 0 0];
    varargout = {[1],[2]};
    return;
end

X = [];

subS = VAR_IN{1};

r1 = is_relu( subS(1) );
r2 = is_relu( subS(2) );
p1 = strcmp( subS(1).type{1}, 'PReLU' );
p2 = strcmp( subS(2).type{1}, 'PReLU' );

if ~( (r1 || p1) && (r2 || p2) ), return; end

if r1
    X = subS(1);
    X.top = subS(2).top;
elseif r2
    X = subS(2);
    X.bottom = subS(1).bottom;
else
    X = subS(1);
    X.top = subS(2).top;    
end

function a = is_relu( l )

a = strcmp( l.type{1}, 'ReLU' ) && isempty( try_to_eval( 'l.relu_param', [] ) );
