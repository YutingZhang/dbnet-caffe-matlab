function varargout = tic_print( varargin )

global TIC_PRINT_TS

if ~isempty(varargin)
    fprintf(varargin{:});
end
TIC_PRINT_TS = tic;

if nargout>=1
    varargout{1} = TIC_PRINT_TS;
end

