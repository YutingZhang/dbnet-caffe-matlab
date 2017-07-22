function varargout = try_fetchOutputs( E )

VAROUT = cell(1,E.numOutputs);
is_got = false;
if ~isempty(E.F)
    try
        wait(E.F);
        [VAROUT{:}] = fetchOutputs(E.F);
        is_got = true;
    catch
        warning( 'cannot fetch outputs, do it in a sync way' );
    end
end
if ~is_got
    [VAROUT{:}] = E.func(E.ARGIN{:});
end

varargout = VAROUT;

