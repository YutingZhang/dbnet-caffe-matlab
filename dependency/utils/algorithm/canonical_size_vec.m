function s = canonical_size_vec( s, min_dim )

if nargin<2
    min_dim = 0;
end

s = reshape(s, 1, numel(s));
for k = numel(s):-1:1
    if s(k) ~= 1
        break;
    end
end
s = s(1:k);

if min_dim

    s = [s, ones(1,max(0,min_dim-numel(s)))];
        
end

