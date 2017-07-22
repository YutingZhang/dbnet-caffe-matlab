function S = obj2struct( obj )

msgId = 'MATLAB:structOnObject';
winfo = warning('query',msgId);
restoreWstate = onCleanup( ...
    @() warning( winfo.state, msgId ) );

warning('off',msgId);
S = struct(obj);
