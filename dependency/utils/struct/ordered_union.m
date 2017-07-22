function U = ordered_union(ORDER,varargin)

switch ORDER
    case 'rstable'
        U = {};
        for k = 1:length(varargin)
            U = union( varargin{k}, U, 'stable' );
        end
    case 'stable'
        U = {};
        for k = 1:length(varargin)
            U = union( U, varargin{k}, 'stable' );
        end
    case 'sorted'
        U = unique(cat(1,varargin{:}));    
    otherwise
        error( 'Unrecognized ORDER' );
end

end
