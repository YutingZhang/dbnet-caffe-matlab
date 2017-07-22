function S1 = caffeproto_apply_blob_revision( S0 )

br = isfield( S0.layer, 'BOTTOM' );
tr = isfield( S0.layer, 'TOP' );

if ~(br || tr), 
    S1 = S0;
    return;
end

S  = S0;

for k = 1:length(S.layer)
    if br
        r  = S.layer(k).BOTTOM;
        r0 = S.layer(k).bottom;
        if ~isempty(r)
            activeIdxB = ~cellfun( @isempty, r );
            maxActiveIdx = find(activeIdxB, 1, 'last');
            assert( isempty(maxActiveIdx) || maxActiveIdx<=length(r0), ...
                'revision index should not be larger than original dim' );
            S = caffeproto_rename_blob( S, k, r0(activeIdxB), r(activeIdxB), 'b' );
        end
    end
    if tr
        r  = S.layer(k).TOP;
        r0 = S.layer(k).top;
        if ~isempty(r)
            activeIdxB = ~cellfun( @isempty, r );
            maxActiveIdx = find(activeIdxB, 1, 'last');
            assert( isempty(maxActiveIdx) || maxActiveIdx<=length(r0), ...
                'revision index should not be larger than original dim' );
            S = caffeproto_rename_blob( S, k, r0(activeIdxB), r(activeIdxB), 't' );
        end
    end
end

if br, S.layer = rmfield( S.layer, 'BOTTOM' ); end
if tr, S.layer = rmfield( S.layer, 'TOP' ); end

S1 = S;

