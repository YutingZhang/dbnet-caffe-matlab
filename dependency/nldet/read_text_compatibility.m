function A = read_text_compatibility( ...
    query_ids, gallary_ids, compatibility_matrix )

C = compatibility_matrix;
avail_query_idxb   = ( query_ids<=size(C,1) );
avail_gallary_idxb = ( gallary_ids<=size(C,1) );
A = zeros( numel(avail_query_idxb), numel(avail_gallary_idxb) );
A(avail_query_idxb,avail_gallary_idxb) = ...
    C( query_ids(avail_query_idxb), ...
    gallary_ids(avail_gallary_idxb) );

A = max(A, bsxfun( @eq, vec(query_ids), vec(gallary_ids).' ) );
