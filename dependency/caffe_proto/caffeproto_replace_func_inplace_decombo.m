function X = caffeproto_replace_func_inplace_decombo( varargin )

if strcmp( varargin{1}, 'extend' )
    X = {};
    return;
end

if strcmp( varargin{1}, 'adjacent' )
    X = [0];
    return;
end

X = [];

subS = varargin{1};

if ~strcmp('InplaceCombo',subS.type{1}),
    return;
end

X = subS.layers;

for k = 2:length(X)
    atyp = caffeproto_abbrev_type( X(k) );
    newLayerName0 = [ subS.name{1} '/' atyp{1}];
    existNames = [X(1:k-1).name];
    newLayerName = newLayerName0;
    j = 0;
    while ismember(newLayerName,existNames)
        j = j+1;
        newLayerName = [ newLayerName0 sprintf('[%d]',j)];
    end
    X(k).name = {newLayerName};
end
