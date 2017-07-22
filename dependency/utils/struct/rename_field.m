function S1=rename_field(S0, oldName, newName)

if iscell( oldName )
    assert( iscell(newName) && numel(oldName)==numel(newName), ...
        'oldName and newName should be matched in dimesion' );
    S1 = S0;
    for k = 1:numel(oldName)
        S1 = rename_field(S1,oldName{k},newName{k});
    end
    return;
end

S1 = rmfield( S0, oldName );
eval( sprintf('S1.%s = S0.%s;',newName,oldName) );


