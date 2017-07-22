function [S, cellmerge_dim ] = struct_cell2struct( ...
    S, recursive_depth, numeric2double )
% S = struct_cell2struct( S, recursive_depth, double4numeric )

if nargin<2 || isempty(recursive_depth)
    recursive_depth = inf;
end
if nargin<3
    numeric2double = 0;
end

cellmerge_dim = 0;

if recursive_depth < 0
    return;
end

if isstruct(S)
    if recursive_depth>0
        for k=1:numel(S)
            S(k) = structfun( ...
                @(a) struct_cell2struct( a, recursive_depth-1, numeric2double ), S(k), ...
                'UniformOutput', 0 );
        end
    end
elseif iscell(S)
    if recursive_depth>0
        [S, pre_ccd] = cellfun( ...
            @(a) struct_cell2struct( a, recursive_depth-1, numeric2double ), S, ...
            'UniformOutput', 0);
        pre_ccd = cell2mat(pre_ccd);
        pre_ccd = unique(pre_ccd);
    else
        pre_ccd = 0;
    end
    
    if isscalar(pre_ccd)
        C = cellfun( @class, S(:), 'UniformOutput', 0 );
        uC = unique( C );
        if isscalar( uC ) 
            sz_vec = cellfun( @size, S(:), 'UniformOutput', 0 );
            sz_len = cellfun( @length, sz_vec );
            u_sz_len = unique(sz_len);
            if isscalar(u_sz_len)
                u_sz = unique( cat( 1, sz_vec{:} ), 'rows' );
                if size(u_sz,1) == 1
                    if u_sz_len == 2
                        if all( u_sz == [1, 1] )
                            cat_dim = 1;
                        elseif u_sz(2) == 1
                            cat_dim = 2;
                        else
                            cat_dim = 3;
                        end
                    else
                        cat_dim = u_sz_len + 1;
                    end
                    cat_dim = max(pre_ccd+1,cat_dim);
                    S1 = [];
                    try
                        S1 = cat( cat_dim, S{:} );
                    catch
                        if strcmp( uC, 'struct' )
                            S1 = cat_struct( cat_dim, S{:} );
                        end
                    end
                    if ~isempty(S1)
                        if isvector(S)
                            s_sz = length(S);
                        else
                            s_sz = size(S);
                        end
                        u_sz = canonical_size_vec(u_sz, cat_dim-1);
                        total_sz = [u_sz(1:(cat_dim-1)), s_sz];
                        S = reshape( S1, canonical_size_vec( total_sz, 2 ) );
                        cellmerge_dim = length(total_sz);
                    end
                end
            end
        end
    end
elseif isnumeric(S)
    if numeric2double
        S = double(S);
    end
end

