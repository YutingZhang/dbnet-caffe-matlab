function S1 = caffeproto_add_ioshape( S0, backendType )

if ~exist('backendType','var') || isempty(backendType)
    backendType = 'matcaffe';
end

switch backendType
    case 'matcaffe'
        S1 = caffeproto_add_ioshape_matcaffe( S0 );
    case 'struct'
        S1 = caffeproto_add_ioshape_struct( S0 );
    otherwise
        error( 'Unknown backendType' );
end



function S1 = caffeproto_add_ioshape_matcaffe( S0 )

[blobShapes,blobNames] = caffeproto_matcaffe_blobshapes( S0 );
blobN=length(blobNames);

S = S0;
for k = 1:length(S.layer)
    if isempty( S.layer(k).bottom )
        S.layer(k).aux.bottom_shapes = {};
    else
        [~,blobIdx] = ismember( S.layer(k).bottom, blobNames );
        blobIdx(blobIdx==0) = blobN+1;
        S.layer(k).aux.bottom_shapes = blobShapes(blobIdx);
    end
    if isempty( S.layer(k).top )
        S.layer(k).aux.top_shapes = {};
    else
        [~,blobIdx] = ismember( S.layer(k).top, blobNames );
        blobIdx(blobIdx==0) = blobN+1;
        S.layer(k).aux.top_shapes = blobShapes(blobIdx);
    end
end

S1 = S;


function S1 = caffeproto_add_ioshape_struct( S0 )

error('not implemented');
S1 = S0;

