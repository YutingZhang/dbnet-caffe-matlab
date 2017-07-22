function toc_print( varargin )

global TIC_PRINT_TS

if ~isempty(varargin) && isnumeric(varargin{1})
    PVAR = varargin(2:end);
    t = varargin{1};
else
    PVAR = varargin;
    t = TIC_PRINT_TS;
end

if ~isempty(PVAR)
    fprintf(PVAR{:});
end

toc(t);

