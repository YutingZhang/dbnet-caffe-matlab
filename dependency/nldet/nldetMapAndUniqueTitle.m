function T1 = nldetMapAndUniqueTitle( T0, newTextIds, mergeMethod )
% T1 = nldetMapAndUniqueTitle( T0, newTextIds, mergeMethod )

if isempty(T0)
    T1 = T0;
    return;
end

if nargin<2
    newTextIds = [];
end

if nargin<3 || isempty(mergeMethod)
    mergeMethod = 'min';
end

switch mergeMethod
    case 'min'
        useMax = 0;
    case 'max'
        useMax = 1;
end

assert( isstruct(T0), 'Unsupported input' );
if isscalar(T0)
    T1 = nldetMapAndUniqueTitle_Single( T0, newTextIds, useMax );
else
    T1 = nldetMapAndUniqueTitle_Multiple( T0, newTextIds, useMax );
end

function T1 = nldetMapAndUniqueTitle_Single( T0, newTextIds, useMax )


text_ids = T0.text_id;
if ~isempty(newTextIds)
    text_ids = newTextIds(text_ids);
end

[text_ids, ~, ic] = unique(text_ids);
labels = zeros(size(text_ids));

if useMax
    for k = 1:numel(text_ids)
        labels(k) = max( T0.label(ic==k) );
    end
else
    for k = 1:numel(text_ids)
        labels(k) = min( T0.label(ic==k) );
    end
end

T1 = T0;
T1.text_id = text_ids;
T1.label   = labels;


function T1 = nldetMapAndUniqueTitle_Multiple( T0, newTextIds, useMax )

if useMax
    merge_func = @max;
else
    merge_func = @min;
end

T = cat(1,T0.text_id);
N = vec( cellfun( @numel, {T0.text_id}.') );
L = cat(1,T0.label);

if ~isempty(newTextIds)
    T = newTextIds(T);
end
[U, ~, uT] = unique(T);
s = length(U); % size of text_id vocabulary

m = numel(N); % number of regions
% N = vec( arrayfun( @(a,b) numel(a.text_id), T0) );

%I = arrayfun(@(a,b) repmat(a,b,1),(1:numel(N)).',N, 'UniformOutput', 0);
%I = cat(1,I{:});
I = RunLength((1:numel(N)).',N);

mergedL = accumarray( [uT,I], L, [s,m], merge_func, inf );

%T1 = repmat(struct('text_id',[],'label',[]),size(T0));
T1 = T0;
V = ~isinf(mergedL); % valid indice
for k = 1:m
    v_k = V(:,k);
    T1(k).text_id = U(v_k,1);
    T1(k).label   = mergedL(v_k,k);
end

%{
V = ~isinf(vec(mergedL)); % valid indice
[ut_idx, rgn_idx] = ind2sub( [s,m], find(V) );

outT = U(ut_idx);
outL = mergedL(V);

z = zeros(0,1);
outT1 = accumarray( rgn_idx, outT, [m 1], @(varargin) {cat(1, z, varargin{:})}, {z} );
outL1 = accumarray( rgn_idx, outL, [m 1], @(varargin) {cat(1, z, varargin{:})}, {z} );

T1 = struct('text_id', outT1, 'label', outL1);
T1 = reshape( T1, size(T0) );
%}
