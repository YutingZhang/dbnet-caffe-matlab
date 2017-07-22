function S = scalar_struct( varargin )

if ~isempty(varargin)
    if isstruct(varargin{1})
        S = varargin{1};
    else
        P = varargin;
        P(2:2:end) = arrayfun( @(a) a, P(2:2:end), 'UniformOutput' , 0 );
        S = struct( P{:} );
    end
else
    S = struct();
end
