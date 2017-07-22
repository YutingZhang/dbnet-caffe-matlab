function assignhere( var_name, val )

tmp_name = 'ASSIGN_HERE_TMP_VAR_____';
try
    assignin( 'caller', tmp_name, val );
catch e
    evalin( 'caller', sprintf('clear %s;', tmp_name ) );
    throw(e);
end

try
    evalin( 'caller', sprintf('%s = %s; clear %s;', var_name, tmp_name, tmp_name ) );
catch e
    evalin( 'caller', sprintf('clear %s;', tmp_name ) );
    throw(e);
end
