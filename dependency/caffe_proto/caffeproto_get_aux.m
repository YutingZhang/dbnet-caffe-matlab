function A = caffeproto_get_aux( S, enable_loop )

if ~exist('enable_loop','var') || isempty(enable_loop)
    enable_loop = false;
end

S = S.layer;
n = numel(S);
A = repmat( struct( 'preBlobs', [], 'nextBlobs', [], ...
    'preLayers', [], 'nextLayers', [] ), n+2, 1 );
topIdx = n+1; bottomIdx = n+2;

% add input and output layers
T0 = unique([ S.top ],'stable');
B0 = unique([ S.bottom ],'stable');
B = cell( size(S) ); % remove inplace
T = cell( size(S) ); % remove inplace
for k = 1:numel(B)
    if isempty( S(k).bottom )
        B{k} = {};
    elseif isempty( S(k).top )
        B{k} = S(k).bottom;
    else
        B{k} = setdiff( S(k).bottom, S(k).top );
    end
    if isempty( S(k).top )
        T{k} = {};
    elseif isempty( S(k).bottom )
        T{k} = S(k).top;
    else
        T{k} = setdiff( S(k).top, S(k).bottom );
    end
end
B = unique([ B{:} ],'stable');
T = unique([ T{:} ],'stable');
inputBlobs = setdiff( B0, T, 'stable' );
outputBlobs= setdiff( T0, B, 'stable' );
S(topIdx).bottom = outputBlobs;
S(topIdx).name = {'[OUTPUT]'};
S(topIdx).type = {'[OUTPUT]'};

S(bottomIdx).top  = inputBlobs;
S(bottomIdx).name = {'[INPUT]'};
S(bottomIdx).type = {'[INPUT]'};


% all top
[nameT,layerT,eltidxT] = cat1dim( { S.top }, 2); 
nameT = nameT{1}; layerT = layerT{1}; eltidxT = eltidxT{1};
% all bottom
[nameB,layerB,eltidxB] = cat1dim( { S.bottom }, 2); 
nameB = nameB{1}; layerB = layerB{1}; eltidxB = eltidxB{1};

% handle duplicated layer (due to inplace layer)
%  currenlty don't consider recurrent case (Caffe also doesn't support it)
nameUniqBT = unique( [nameT, nameB] );

[nameUniqT,~,icT] = unique( nameT );
idxDupT = accumarray( icT, 1:length(icT), [], @(x){x} );
idxDupT = cellfun(@sort,idxDupT,'UniformOutput',false);
numDupT = cellfun(@length,idxDupT);
for k = vec(find(numDupT>1)).'
    r = 1;
    for j = 1:numDupT(k)
        t = idxDupT{k}(j);
        l = layerT(t);
        if ~ismember( nameT{t}, nameB(layerB==l) ) || ismember(nameT{t},inputBlobs)
            % only active for replacement layer (this is safe for mix phases)
            continue;
        end
        
        [newName, r] = find_avaible_name( nameUniqT{k}, r, nameUniqBT );
        
        bidx = find( layerB>l );
        idxbInB = cellfun( @(a) strcmp(nameT{t},a), nameB(bidx) );
        nameB(bidx(idxbInB)) = {newName};
        
        nameT(idxDupT{k}(j:end)) = {newName};
        
        nameUniqBT = [{newName}, nameUniqBT];
    end
end

% link blobs
[~,forwardBlobLink]  = ismember_dup(nameT,nameB);
[~,backwardBlobLink] = ismember_dup(nameB,nameT);

% map links to layers

if enable_loop
    cleanNext_func = @(a,layerId) a;
    cleanPre_func  = @(a,layerId) a;
else
    cleanNext_func = @(a,layerId) a( :, a(1,:)>layerId & a(1,:)~=bottomIdx );
    cleanPre_func  = @(a,layerId) a( :, a(1,:)<layerId | a(1,:)==bottomIdx );
end

for k = 1:n+2
    % forward
    idxbLayerT_k = (layerT == k);
    idxLinkedBottomBlob_k = forwardBlobLink(idxbLayerT_k);
    A(k).nextBlobs  = cellfun( ...
        @(a) [layerB(a); eltidxB(a)], ... 
        idxLinkedBottomBlob_k,'UniformOutput',false);
    A(k).nextBlobs = cellfun(@(a) cleanNext_func(a,k), ...
        A(k).nextBlobs, 'UniformOutput',false);
    if ~isempty(A(k).nextBlobs)
        allForwardLinks_k = [A(k).nextBlobs{:}];
        if ~isempty(allForwardLinks_k)
            A(k).nextLayers = unique( allForwardLinks_k(1,:) );
            A(k).nextLayers = cleanNext_func(A(k).nextLayers,k);
        else
            A(k).nextLayers = [];
        end
    end
    % backward
    idxbLayerB_k = (layerB == k);
    idxLinkedTopBlob_k = backwardBlobLink(idxbLayerB_k);
    A(k).preBlobs  = cellfun( ...
        @(a) [layerT(a); eltidxT(a)], ... 
        idxLinkedTopBlob_k,'UniformOutput',false);

    A(k).preBlobs = cellfun(@(a) cleanPre_func(a,k), ...
        A(k).preBlobs, 'UniformOutput',false);

    if ~isempty(A(k).preBlobs)
        allBackwardLinks_k = [A(k).preBlobs{:}];
        if ~isempty(allBackwardLinks_k)
            A(k).preLayers = unique( allBackwardLinks_k(1,:) );
            A(k).preLayers = cleanPre_func(A(k).preLayers,k);
        else
            A(k).preLayers = [];
        end
    end
end

% make final output

seqIdx = num2cell(1:length(A));
[A.idx] = deal(seqIdx{:});

A = struct('layer',A,'bottomLayerIdx', bottomIdx, 'topLayerIdx', topIdx);

% % bugfix
% if isempty( A.layer(bottomIdx).nextLayers )
%     A.layer(bottomIdx).nextBlobs = cell(1,0);
% end
% if isempty( A.layer(topIdx).preLayers )
%     A.layer(topIdx).preBlobs = cell(1,0);
% end

end

function [newName, r] = find_avaible_name( baseName, r, poolNames )

newName = [baseName ':' int2str(r)];
while ismember( newName, poolNames )
    r = r+1;
    newName = [baseName ':' int2str(r)];
end

end

