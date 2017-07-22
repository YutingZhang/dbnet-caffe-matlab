function S = merge_struct(varargin)
% S = struct_merge(S1,S2,S3,...)

assert( all(cellfun(@isstruct, varargin)), 'All inputs should be struct' )

N = length(varargin);
F = cellfun(@(a) {fieldnames(a)},varargin);

allF = {};
for k = 1:N
    allF = union(allF,F{k});
end

if length(allF)~=sum(cellfun(@length,F))
    error( 'Conflict fields' );
end

clear allF

C = cell(1,N);
for k = 1:N
    C_k  = cellfun( @(a) {sprintf(' ''%s'', reshape({varargin{%d}.%s},size(varargin{%d})),',a,k,a,k)}, F{k} );
    C{k} = cell2mat(C_k.');
end
C(cellfun( @isempty, C )) = {''};
C = cell2mat(C);
C = sprintf('S=struct(%s );',C(1:end-1) );
eval(C);

end
