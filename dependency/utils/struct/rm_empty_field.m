function S1 = rm_empty_field( S0, recursive_depth, is_empty_func )

if ~exist('recursive_depth','var') || isempty(recursive_depth)
    recursive_depth = 1;
end

if ~exist('is_empty_func','var') || isempty(is_empty_func)
    is_empty_func = @(a) isempty(a) && isnumeric(a);
end

if recursive_depth<1
    S1 = S0;
    return;
end

S = S0;
F = fieldnames(S0);
vidxb = true(1,length(F));

for k = 1:length(F)
    A = eval( sprintf('{S.%s}', F{k}) );
    if all( cellfun( is_empty_func, A ) )
        vidxb(k) = false;
    else
        if recursive_depth>0
            for j = 1:length(A)
                if isstruct(A{j})
                    B_j = rm_empty_field( A{j}, recursive_depth-1 );
                    eval( sprintf('S(j).%s=B_j;',F{k}) );
                end
            end
        end
    end
end

S1 = partial_struct( S, F{vidxb} );

