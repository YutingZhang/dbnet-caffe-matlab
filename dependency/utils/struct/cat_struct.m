function A = cat_struct( dimIdx, varargin )

S = varargin;

if isempty( S )
    A = struct();
    return;
end

fldNameArr = cellfun( @fieldnames, S, 'UniformOutput', false );
fldNames   = unique( cat(1,fldNameArr{:}), 'stable' );

outputAug = fldNames;
outputAug(:,2) = {[]}; outputAug = outputAug.';
unitA = struct(outputAug{:});

empty_idxb = cellfun(@isempty,S);
S( empty_idxb ) = [];
fldNameArr( empty_idxb ) = [];

if isempty(S)
    A = repmat(unitA,[0 0]);
    return;
end

sizOther = size(S{1});
sizOther(dimIdx)=0;
for k = 2:length(S)
    sizOther_k = size(S{k});
    sizOther_k(dimIdx) = 0;
    assert( isequal(sizOther_k,sizOther), 'dimension mismatched' );
end

idvLength = cellfun( @(a) size(a,dimIdx), S );

sizA = sizOther;
sizA(dimIdx) = sum(idvLength);

blockSub = arrayfun( @(n) 1:n, sizA, 'UniformOutput', false );

EN = cumsum( idvLength );
ST = [1, EN(1:end-1)+1];

A = repmat(unitA,sizA);

for k = 1:length(S)
    blockSub{dimIdx} = ST(k):EN(k);
    blockAug = outputAug;
    [~,idxBlockFN] = ismember(fldNameArr{k},fldNames);
    for j = 1:length(idxBlockFN)
        blockAug{2,idxBlockFN(j)} = reshape( eval( sprintf( '{ S{k}.%s }', ...
            fldNameArr{k}{j} ) ), size(S{k}) );
    end
    A(blockSub{:}) = struct(blockAug{:});
end
