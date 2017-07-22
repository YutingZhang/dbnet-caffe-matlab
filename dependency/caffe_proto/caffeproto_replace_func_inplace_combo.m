function [X, varargout] = caffeproto_replace_func_inplace_combo( varargin )

if strcmp( varargin{1}, 'extend' )
    X = { {'list', 'iterative', ...
        [{@(varargin) caffeproto_replace_func_inplace_combo( 'internal', varargin{1:end} )}, varargin{end}] } };
    return;
end

assert( strcmp(varargin{1},'internal'), 'wrong branch' );

VAR_IN = varargin(2:end);

if strcmp( VAR_IN{1}, 'extend' )
    X = {};
    return;
end

if strcmp( VAR_IN{1}, 'adjacent' )
    X = [0 1; 0 0];
    varargout = {[1],[2]};
    return;
end


subS = VAR_IN{1};

X = [];

isValid = isscalar( subS(2).bottom ) && ...
    isequal(subS(2).bottom,subS(2).top) && ...
    isscalar( subS(1).top );
if ~isValid, return; end

if ~strcmp( subS(1).type{1}, 'InplaceCombo' )
    X1 = struct();
    X1.name = subS(1).name;
    X1.type = {'InplaceCombo'};
    X1.bottom = subS(1).bottom;
    X1.top  = subS(1).top;
    X1.layers = subS(1);
    X1 = transfer_field(X1,subS(1),'aux');
else
    X1 = subS(1);
end

if ~strcmp( subS(2).type{1}, 'InplaceCombo' )
    X2 = struct();
    X2.name = subS(2).name;
    X2.type = {'InplaceCombo'};
    X2.bottom = subS(2).bottom;
    X2.top  = subS(2).top;
    X2.layers = subS(2);
    X2 = transfer_field(X2,subS(2),'aux');
else
    X2 = subS(2);
end

X = X1;
X.layers = cat_struct(2,X1.layers,X2.layers);
X.top = X2.top;
X.aux.bottom_shapes = try_to_eval( 'X1.aux.bottom_shapes' );
X.aux.top_shapes = try_to_eval( 'X2.aux.top_shapes' );
