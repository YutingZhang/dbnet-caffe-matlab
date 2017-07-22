function [A, cellIdx, eltIdx] = cat1dim( C, dim )

if ~exist('dim','var')
    dim = 1;
end

s = size(C);

if length(s)<dim
    s = [s,ones(1,dim-length(s))];
end
if s(dim)<1
    s1 = s;
    s1(dim) = 1;
    A = cell(s1);
    cellIdx = cell(s1);
    eltIdx  = cell(s1);
    return;
end

L = arrayfun( @(a) {ones(a,1)}, s );
L{dim} = s(dim);

A = mat2cell( C, L{:} );
A = cellfun( @(a) {cat(dim,a{:})}, A );

if nargout>=2
    t = s; t(dim)=1;
    u = ones(size(s)); u(dim) = s(dim);
    B = cellfun( @(c) size(c,dim), C );
    I = repmat( reshape(1:s(dim),u), t );
    cellIdx = arrayfun( @(b,i) {repmat(i,set_ith_elt( u, dim, b ))}, B, I );
    cellIdx = cat1dim( cellIdx, dim );
    eltIdx  = arrayfun( @(b,i) {reshape(1:b, set_ith_elt( u, dim, b ))}, B, I );
    eltIdx = cat1dim( eltIdx, dim );
end

function a = set_ith_elt( a, i, v )
a(i) = v;

