function S1 = caffeproto_rename_blob( S0, layerId, ...
    srcBlobNames, dstBlobNames, effectiveRange )
% S1 = caffeproto_rename_blob( S0, layerId, srcBlobName, dstBlobNames, effectiveRange )
% effectiveRange: 
%   't',  'top'     : rename top, and locally resolve inplace
%   't+', 'top+'    : rename top, and globally resolve inplace
%   'b',  'bottom'  : rename bottom, and locally resolve inplace
%   'b+', 'bottom+' : rename bottom, and globally resolve inplace

if ischar(srcBlobNames), srcBlobNames = {srcBlobNames}; end
if ischar(dstBlobNames), dstBlobNames = {dstBlobNames}; end

assert( numel(srcBlobNames) == numel(dstBlobNames), 'src and dst should have the same numel' );

srcBlobNames = vec(srcBlobNames);
dstBlobNames = vec(dstBlobNames);

same_idxb = cellfun( @strcmp, srcBlobNames,dstBlobNames);
srcBlobNames = srcBlobNames(~same_idxb);
dstBlobNames = dstBlobNames(~same_idxb);

A = [];
if iscell(S0),
    assert( numel(S0)==2, 'Unrecognized S0' );
    A = S0{2};
    S0 = S0{1};
end
    
if isempty(srcBlobNames)
    S1 = S0;
    return;
end

S = S0;

layerN = numel(S.layer);

assert(layerId>0 && layerId<=layerN, 'Wrong layerId');

is_global = (effectiveRange(end)=='+');
tb = lower( effectiveRange(1) );
assert( ismember(tb, {'t','b'}) );

if is_global
    if tb == 't'
        S.layer(layerId).top = map_with_codebook( S.layer(layerId).top, ...
            [srcBlobNames, dstBlobNames] );
        S = caffeproto_replace( S, 'subnet', (layerId+1):length(S.layer), ...
                {'rename_blob', srcBlobNames, dstBlobNames } );
    else
        S.layer(layerId).bottom = map_with_codebook( S.layer(layerId).bottom, ...
            [srcBlobNames, dstBlobNames] );
        S = caffeproto_replace( S, 'subnet', 1:(layerId-1), ...
                {'rename_blob', srcBlobNames, dstBlobNames } );
    end
else
    if isempty(A)
        A = caffeproto_get_aux( S0 );
    end
    for k = 1:length( srcBlobNames )
        S = caffeproto_rename_blob_local( ...
            S, A, layerId, tb, srcBlobNames{k}, dstBlobNames{k} );
    end
end

S1 = S;

function S = caffeproto_rename_blob_local( ...
    S, A, layerId, tb, srcBlobName, dstBlobName )

layerN = numel(S.layer);

k = layerId;
if tb == 't'
    while k<layerN && ~isempty(S.layer(k).top)
        matched_idxb = ismember(S.layer(k).top, {srcBlobName});
        if ~any(matched_idxb), break; end
        assert( sum(matched_idxb)<=1, 'mutliple top blobs have the same name' );
        S.layer(k).top{matched_idxb} = dstBlobName;
        nb = A.layer(k).nextBlobs{matched_idxb};
        for j = size(nb,2)
            if nb(1,j)<=layerN
                S.layer(nb(1,j)).bottom{nb(2,j)} = dstBlobName;
            end
        end
        if size(nb,2)==1 % only support sequential track
            k = nb(1);
        else
            break;
        end
    end
else
    while k<layerN && ~isempty(S.layer(k).bottom)
        matched_idxb = ismember(S.layer(k).bottom, {srcBlobName});
        if ~any(matched_idxb), break; end
        S.layer(k).bottom(matched_idxb) = { dstBlobName };
        matched_idx = find(matched_idxb,1);
        pb = A.layer(k).preBlobs{matched_idx};
        for j = size(pb,2)
            lid = pb(1,j);
            tid = pb(2,j);
            assert( size( A.layer(lid).nextBlobs{tid}, 2 )==1, ...
                'Not implemented' ); % add split layers
            if pb(1,j)<=layerN
                S.layer(lid).top{tid} = dstBlobName;
            end
        end
        if size(pb,2)==1 % only support sequential track
            k = pb(1);
        else
            break;
        end
    end
end
