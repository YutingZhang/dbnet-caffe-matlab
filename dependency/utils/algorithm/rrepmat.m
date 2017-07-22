function B = rrepmat( A, target_siz )
% rrepmat( A, target_siz );

assert( ~isempty(A), 'A cannot be empty' );

if isscalar( target_siz ), 
    target_siz = [target_siz,target_siz]; 
end

target_siz = reshape(target_siz, 1, numel(target_siz) );
a_siz = size(A);

m_siz = max( length(target_siz), length(a_siz) );
sizT = ones( 1, m_siz, 'like', target_siz ); sizT(1:length(target_siz)) = target_siz;
sizA = ones( 1, m_siz, 'like', target_siz ); sizA(1:length(a_siz)) = a_siz;
cycN = ceil( sizT./sizA );

B = repmat(A,cycN);
if isa( A, 'gpuArray' )
    logical_type = 'gpuArray';
else
    logical_type = 'logical';
end

b_ind = cell(1,numel(sizT));
for k = 1:numel(b_ind)
    b_ind{k} = true(sizT(k),1, logical_type);
end
%B = subsref( B, substruct('()', b_ind) );
B = B(b_ind{:});