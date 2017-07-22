function donotcare = nldet_donotcare_from_gt_and_textsim( text_compatible, is_gt )

assert( ismatrix(text_compatible), ...
    'text_compatible should be a matrix' );

numRefText = size(text_compatible,1);

assert( numRefText == size(is_gt,2), ...
    'number of reference text mismatched' );

C = bsxfun( @and, is_gt, shiftdim( text_compatible, -1 ) );

donotcare = reshape( any( C, 2 ), size(is_gt,1), size(text_compatible,2) );
donotcare = donotcare & ~is_gt; % gt itself should not be donotcare
