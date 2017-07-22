function SUB_DICT = codebook_cols( CODE_BOOK, COL_NAMES )
% extract cols for specific name

if ~iscell(COL_NAMES)
    COL_NAMES = {COL_NAMES};
end

[in_codebook, col_idx] = ismember( COL_NAMES, CODE_BOOK.cols );

assert(all(in_codebook), 'COL_NAME does not exist' );

SUB_DICT = CODE_BOOK.dict(:,col_idx);

