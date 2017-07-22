function [B, is_dec] = map_with_codebook( A, CODE_BOOK )
% B = map_with_codebook( A, CODE_BOOK )
% CODE_BOOK: N x 2

if isempty(A) || isempty(CODE_BOOK),
    B = A;
    is_dec = false( size(A) );
    return;
end

c2c = ~iscell(A) && iscell(CODE_BOOK);
if c2c
    A = {A};
end

[is_dec, dec_loc] = ismember(A(:),CODE_BOOK(:,1));
B = A;
B(is_dec) = CODE_BOOK(dec_loc(is_dec),2:end);

if c2c
    B = B{:};
end
