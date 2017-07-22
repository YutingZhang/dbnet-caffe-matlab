function S = empty_struct( varargin )
% Usage: S = empty_struct( S0 )
%        S = empty_struct( FIELD_NAME1, FIELD_NAME2, ... )

if nargin==1 && isstruct( varargin{1} )
    F = fieldnames(varargin{1});
else
    F = varargin;
end

if isempty(F)
    S = struct();
    return;
end

CMD = cellfun( @(a) {sprintf(' ''%s'',[],', a)} , F );
CMD = [ CMD{:} ];
S = eval( sprintf('struct(%s )', CMD(1:end-1)) );

end
