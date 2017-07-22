function [blob, valid_length] = character_enc( phrases, alphabet, ...
    padding_character, output_length )
% [blob, valid_length] = character_enc( phrases, alphabet, output_length )


if ischar(phrases)
    phrases = {phrases};
end

if nargin<3
    padding_character = []; % empty for periodic
end

phrases = vec(phrases).';

num_phrases = length(phrases);

valid_length = cellfun( @length, phrases );
%if exist('output_length', 'var') && ~isempty(output_length)
if nargin>=4 && ~isempty(output_length)
    valid_length = min( valid_length, output_length );
else
    output_length = max(valid_length);
end

if isempty(padding_character)
    C = repmat( ' ', num_phrases, output_length );
    for k = 1:num_phrases
        phrase_k = phrases{k};
        if ~isempty( phrase_k )
            pe = phrase_k(end);
            if (pe>='a' && pe<='z') || (pe>='A' && pe<='Z')
                phrase_k = [phrase_k, ' .'];
            end
            C(k,:) = rrepmat( [phrase_k ' '], [1, output_length] );
        end
    end
else
    assert( ischar(padding_character) && isscalar(padding_character), ...
        'padding_character should be a char scalar' );
    C = repmat( padding_character, num_phrases, output_length );
    for k = 1:num_phrases
        C(k,1:valid_length(k)) = phrases{k};
    end
end

use_gpu = isa(alphabet,'gpuArray');

if use_gpu
    C0   = C;
    C   = zeros(size(C0),classUnderlying(alphabet));
    C(:) = C0(:);
    C = gpuArray(C);
    
    I = zeros(numel(C),2,'like',gpuArray(uint64([])));
    I(:,2) = gpuArray(1):gpuArray(numel(C));
    
    unknownFallbackChar = zeros([1 1],classUnderlying(alphabet));
    unknownFallbackChar(:) = ' ';
    unknownFallbackChar = gpuArray(unknownFallbackChar);
    
    indicatorUnit = gpuArray(single(1));
else
    I = zeros(numel(C),2,'uint64');
    I(:,2) = 1:numel(C);
    unknownFallbackChar = ' ';
    
    indicatorUnit = single(1);
end

[in_alphabet, I(:,1)] = ismember(C(:), alphabet);
I(~in_alphabet,1)  = ismember(unknownFallbackChar, alphabet);

alphabet_length = length(alphabet);

B = accumarray( I, indicatorUnit, [alphabet_length, numel(C)], [] );

%B = sparse( I, colIdx, 1, alphabet_length, numel(I) );
%B = single(full(B));

B = reshape( B, [1, alphabet_length, size(C)] ); %[1, alphabet, batch, output_length]
B = shiftdim( B, 3 ); % [1, output_length, 1, alphabet, batch]

blob = B;

