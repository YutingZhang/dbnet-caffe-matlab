function X = caffeproto_replace_func_remove_dup_branch( varargin )

if strcmp( varargin{1}, 'extend' )
    X = { {'list', 'iterative', ...
        [{@(varargin) caffeproto_replace_func_remove_dup_branch( 'internal', varargin{1:end} )}, varargin{end}] } };
    return;
end

assert( strcmp(varargin{1},'internal'), 'wrong branch' );

VAR_IN = varargin(2:end);

if strcmp( VAR_IN{1}, 'extend' )
    X = [];
    return;
end

if strcmp( VAR_IN{1}, 'adjacent' )
    X = [0 1 1; 0 0 0; 0 0 0];
    return;
end

X = [];

subS = VAR_IN{1};

is_matched = ( numel(subS(2).top) == numel(subS(3).top) ) && isequal( ...
    rm_empty_field( partial_struct( subS(2), '@exclude', 'top', 'name' ) ), ...
    rm_empty_field( partial_struct( subS(3), '@exclude', 'top', 'name' ) ) ...
    );
if ~is_matched, return; end

X = subS(1:2);
X(2).top = subS(3).top;
X(2).TOP = subS(2).top;

