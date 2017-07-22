function v = default_eval( expStr, defVal, invalidCond )

if ~exist( 'INVALID_COND', 'var' ) || isempty( invalidCond )
    invalidCond = @(a) isnumeric(a) && isempty(a);
end

try
    v = evalin( 'caller', expStr );
catch 
    v = defVal;
end

if invalidCond(v), v = defVal; end
