function S1 = caffeproto_make_deploy( S0, batch_size, input_force_backward )

%
if ~exist('batch_size','var') || isempty(batch_size)
    batch_size = 1;
end

if ~exist('input_force_backward','var') || isempty(input_force_backward)
    input_force_backward = 0;
end

S = S0;
% remove data layer

S = caffeproto_replace( S, {'rm_if', ...
    @(a) ~isempty( strfind(a.type{1}, 'Data') ) } );


% remove inplace top
TC = { S.layer.top };
BC = { S.layer.bottom };

T = cellfun(@(t,b) setdiff(t,b,'stable'), TC,BC, 'UniformOutput', 0);
T = cat(2,T{cellfun(@(a) ~isempty(a),T)});

% find input 

%T = unique( cat( 2, S.layer.top ), 'stable' );
B = unique( cat( 2, S.layer.bottom ), 'stable' );
I = setdiff( B,T, 'stable' );

if isempty(I), 
    S1 = S;
    return; 
end

% remove input linked only to SilenceLayer
inputValidity = false(size(I));
for k = 1:length(S.layer)
    if ~strcmp(S.layer(k).type{1},'Silence') && ~isempty(S.layer(k).bottom)
        inputValidity = inputValidity | ismember( I, S.layer(k).bottom );
    end
end

U = I(~inputValidity);
I = I(inputValidity);
% reorder input according to original network definition
defI = caffeproto_defined_inputs(S);
[is_defI, pos_defI]= ismember(I,defI);
I=[I(pos_defI(is_defI)),I(~is_defI)];

% check validity
numInputs = length(I);
if isscalar( batch_size )
    batch_size = repmat(batch_size, 1, numInputs);
else
    assert( isvector(batch_size) && length(batch_size) == numInputs, ... 
        'dim of batch_size mismatch with num of inputs' );
end

for k = 1:length(S.layer)
    S.layer(k).bottom = setdiff( S.layer(k).bottom, U, 'stable' );
    if isempty( S.layer(k).bottom ) && strcmp(S.layer(k).type{1},'Silence')
        S.layer(k).type = [];
    end
end

S.layer( cellfun( @isempty, {S.layer.type} ) ) = [];

% get blob size
if ~isfield(S.layer,'aux')
    S0 = caffeproto_add_ioshape( S0, 'matcaffe' );
end


% add input blobs
inputBlobs = struct( 'input', {[]}, 'input_shape', {[]} );
for j = 1:numInputs
    
    % figure out the blob size
    blobShape = [];
    for k = 1:length(S0.layer)
        [is_in_bottom, bottom_id] = ismember( I{j}, S0.layer(k).bottom );
        if is_in_bottom
            blobShape = S0.layer(k).aux.bottom_shapes{bottom_id};
            break;
        end
        [is_in_top, top_id] = ismember( I{j}, S0.layer(k).top );
        if is_in_top
            blobShape = S0.layer(k).aux.top_shapes{top_id};
            break;
        end
    end
    
    % add input blob
    inputBlobs.input{j} = I{j};
    inputBlobs.input_shape(j).dim = [ batch_size(j), blobShape ];
    
end

if islogical( input_force_backward ) || isnumeric(input_force_backward)
    input_force_backward = boolean(input_force_backward);
    if isscalar( input_force_backward )
        input_force_backward = repmat( input_force_backward, ...
            1, numel(inputBlobs.input_shape) );
    end
else
    assert( iscell(input_force_backward), 'Cannot parse input_force_backward' );
    ifbn = cellfun( @isnumeric, input_force_backward );
    input_force_backward_on_idx = [input_force_backward{ifbn}];
    assert( all(ceil(input_force_backward_on_idx)==input_force_backward_on_idx) && ...
        all(input_force_backward_on_idx>0) && ...
        all(input_force_backward_on_idx<=numel(inputBlobs.input)), ...
        'invalid input idx' );
    input_force_backward_strcell = input_force_backward(~ifbn);
    assert( all(ismember(input_force_backward_strcell,inputBlobs.input)), ...
        'specified input does not exist' );
    input_force_backward = ismember( inputBlobs.input, input_force_backward_strcell );
    input_force_backward(input_force_backward_on_idx) = true;
end

if any( input_force_backward )
    assert( numel(input_force_backward) == numel(inputBlobs.input_shape), ...
        'input force_backward dim mismatched' );
    inputBlobs.input_force_backward = input_force_backward;
end

S1_1 = rmfield(S,'layer');
S1 = xmerge_struct( 'always', 'always', S1_1, inputBlobs );
S1.layer = S.layer;
