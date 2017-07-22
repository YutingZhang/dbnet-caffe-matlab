function [availableIters, availableIters_str] = nldetAvailTrainSnapshotIters( SpecificDir4Train )

if ~exist( 'SpecificDir4Train', 'var' ) || isempty(SpecificDir4Train)
    SpecificDir4Train = evalin( 'caller', 'SPECIFIC_DIRS.Train' );
end

L = dir(SpecificDir4Train);
L(~[L.isdir]) = [];
L = {L.name};

availableIters = -ones( numel(L), 1 );
availableIters_str = cell( numel(L), 1 );

for k = 1:numel(L)
    T = regexp( L{k}, '^iter_([0-9\.]*)$', 'once', 'tokens' );
    if ~isempty(T)
        availableIters(k) = str2double( T{1} );
        availableIters_str{k} = T{1};
    end
end

invalid_idxb = availableIters<0;
availableIters( invalid_idxb )     = [];
availableIters_str( invalid_idxb ) = [];

[availableIters, sidx] = sort(availableIters);
availableIters_str = availableIters_str(sidx);

