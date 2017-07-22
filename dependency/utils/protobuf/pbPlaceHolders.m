function [P, defaultValues] = pbPlaceHolders( S, use_struct_output )
% return place holders in the prototxt files and their default values (if any)

if nargin<2 || isempty(use_struct_output)
    use_struct_output = 0;
end

if isfield(S, 'PlaceHolders__')
    % default value is stored here
    D = S.PlaceHolders__;
    S = rmfield( S, 'PlaceHolders__' );
else
    D = struct();
end

P = check_place_holders(S);
P = unique([P;fieldnames(D)]);

defaultValues = cell(size(P));

for k = 1:numel(P)
    if isfield( D, P{k} )
        defaultValues{k} = subsref( D, substruct('.',P{k}) );
    end
end

if use_struct_output
    scmd = [vec(P),vec(defaultValues)].';
    P = scalar_struct(scmd{:});
    defaultValues = [];
end

function P = check_place_holders( S )

P = {};
if isstruct(S)
    S = struct2cell(S);
    for k = 1:numel(S)
        P1 = check_place_holders( S{k} );
        P = [P;P1];
    end
elseif isa( S, 'proto_enum_class')
    for k = 1:numel(S)
        v = S(k).val;
        if ischar(v) && strncmp( v, '$', 1 )
            P = [P;{v(2:end)}];
        end
    end
end
