function S1 = pbSpecializePlaceHolders( S0, varargin )
% S1 = pbSpecializePlaceHolders( S0, prototxt_string4place_holder_values )
% S1 = pbSpecializePlaceHolders( S0, struct4place_holder_values )
% S1 = pbSpecializePlaceHolders( S0, place_holder1, value1, ... )

if numel(varargin)==1
    T = varargin{1};
    if ischar(T)
        D1 = prototxt2struct( T, 'string' );
    elseif isstruct(T) && isscalar(T)
        D1 = T;
    else
        error( 'Unrecognized place holder specification' );
    end
else
    assert( ceil(numel(varargin)*0.5)*2 == numel(varargin), ...
        'wrong number of arguments' );
    D1 = scalar_struct( varargin{:} );
end

[P,D0c] = pbPlaceHolders( S0 );

D = partial_struct(D1, P{:});
D0s = [vec(P).';vec(D0c).'];
D0 = scalar_struct(D0s{:});
D = xmerge_struct( D0, D );

D = structfun( @(a) vec(a).', D, 'UniformOutput', 0);

S1 = specialize_place_holders(S0, D);
if isfield(S1,'PlaceHolders__')
    S1 = rmfield(S1,'PlaceHolders__');
end


function S1 = specialize_place_holders(S0, D)

if isstruct( S0 )
    S1 = S0;
    for k = 1:numel(S0)
        S1(k) = structfun( @(s) specialize_place_holders(s,D), S0(k), ...
            'UniformOutput', 0 );
    end
elseif isa( S0, 'proto_enum_class')
    S1c = cell(size(S0));
    for k = 1:numel(S0)
        v = S0(k).val;
        if ischar(v) && strncmp( v, '$', 1 )
            ss = substruct( '.', v(2:end) );
            u = subsref(D,ss);
            S1c{k} = u;
        elseif ischar(v) && strncmp( v, '@', 1 )
            mcmd = v(2:end);
            mcmd = strrep(mcmd,'$','D.');
            S1c{k} = eval(mcmd);
        else
            S1c{k} = S0(k);
        end
    end
    try 
        S1 = cell2mat(S1c);
    catch
        try
            S1 = cat(1,S1c{:});
            S1 = reshape(S1,size(S1c));
        catch
            S1 = cat_struct(1,S1c{:});
            S1 = reshape(S1,size(S1c));
        end
    end
else
    S1 = S0;
end
