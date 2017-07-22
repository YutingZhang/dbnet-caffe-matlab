function [snapshotDir, chosenIter] = nldetNearestTrainSnapshot( ...
    iterUpperBound, SpecificDir4Train )

if ~exist( 'SpecificDir4Train', 'var' ) || isempty(SpecificDir4Train)
    SpecificDir4Train = evalin( 'caller', 'SPECIFIC_DIRS.Train' );
end

[availableIters, availableIters_str] = nldetAvailTrainSnapshotIters( SpecificDir4Train );

chosenIter = max( availableIters( availableIters<=iterUpperBound ) );

if isempty(chosenIter)
    snapshotDir = [];
else
    [~,p] = ismember( chosenIter, availableIters );
    snapshotDir = fullfile( SpecificDir4Train, ...
        sprintf( 'iter_%s', availableIters_str{p} ) );
end
