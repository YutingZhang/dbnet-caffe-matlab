function [S, varargout] = var2struct(varargin)
% Usage: S = var2struct(VAR1_NAME,VAR2_NAME,...)
%        S = var2struct(VAR1,VAR2,...)
%        S = var2struct(VAR_NAME_CELL,'-string-cell')
%        S = var2struct(...,'-clear')
%        [S, APPLIED, OMITTED] = var2struct(...,'-fill')
%        [S, APPLIED, OMITTED] = var2struct(...,'-ignore')
%        ... = var2struct(...,'-array');

if nargin==0
    S = struct();
    return;
end

if ismember( '-string-cell', varargin( cellfun(@ischar,varargin) ) )
    scIdx = find( cellfun(@iscell,varargin), 1 );
    if isempty(scIdx)
        error( 'Cannot find a cell' );
    else
        VARS = [reshape(varargin{scIdx},1,numel(varargin{scIdx})), ...
            varargin([1:scIdx-1 scIdx+1:end])];
    end
else
    VARS = varargin;
    for k=1:nargin
        if ~isempty( inputname(k) )
            VARS{k} = inputname(k);
        end
    end
end


assert( all( cellfun( @ischar, VARS ) ), 'All arguments should be string or existing variable' );
assert( ~any( cellfun( @isempty, VARS ) ), 'All arguments should be non-empty' );

need2clear = 0;

opIdxB = cellfun( @(a) a(1)=='-', VARS );

if ~isempty( setdiff( VARS(opIdxB), {'-string-cell','-clear','-fill','-ignore','-array'} ) )
    error( 'Unrecognized options' );
end

array_mode = ismember('-array', VARS(opIdxB) );

if ismember('-clear', VARS(opIdxB) )
    need2clear = 1;
end

unavail_handler = 'error';
use_handler = 0;
if ismember('-ignore', VARS(opIdxB) )
    unavail_handler = 'ignore';
    use_handler = use_handler + 1;
end
if ismember('-fill', VARS(opIdxB) )
    unavail_handler = 'fill';
    use_handler = use_handler + 1;
end
if use_handler>1
    error( 'Conflict options: -ignore and -fill are not compitable' );
end

A = VARS(~opIdxB);
n = length(A);
validIdxB = false(1,n);
for k=1:n
    validIdxB(k) = evalin( 'caller', sprintf('exist(''%s'',''var'')', A{k}) );
end

if array_mode
    data_token = '%s';
else
    data_token = '{%s}';
end
cmd_tmpl = [' ''%s'', ' data_token ','];

switch unavail_handler
    case 'ignore'
        B = A(validIdxB);
        C = cellfun( @(b) {sprintf(cmd_tmpl, b, b)}, B );
    case 'fill'
        B = A;
        B(~validIdxB) = {'[]'};
        C = cellfun( @(a,b) {sprintf(cmd_tmpl , a, b)}, A, B );
    otherwise
        if ~all(validIdxB)
            E = cellfun( @(a) {sprintf(' %s,' , a)}, A(~validIdxB) );
            E = cell2mat(E);
            error( ['I cannot find the variables: ' E(1:end-1)] );
        end
        C = cellfun( @(a) {sprintf(cmd_tmpl , a, a)}, A );
end

C = cell2mat(C);
C = sprintf( 'struct(%s)', C(1:end-1) );
S = evalin( 'caller', C );

if nargout>=1
    varargout{1} = A(validIdxB);
end
if nargout>=2
    varargout{2} = A(~validIdxB);
end

if need2clear
    C = [ 'clear', cell2mat( cellfun( @(a) {sprintf(' %s',a)}, A ) ) ];
    evalin( 'caller', C );
end

end
