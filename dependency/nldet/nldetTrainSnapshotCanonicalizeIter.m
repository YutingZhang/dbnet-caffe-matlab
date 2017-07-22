function iter = nldetTrainSnapshotCanonicalizeIter( iter, PARAM )

if ~ischar(iter), return; end

switch iter
    case 'latest'
        SpecificDir4Train = sysStageSpecificDir( PARAM, 'Train',0,1,1,1);
        avail_iters = nldetAvailTrainSnapshotIters( SpecificDir4Train );
        assert( ~isempty( avail_iters ), ...
            'nldetTrainSnapshotCanonicalizeIter:no_snapshot', ...
            'No snapshot at all' );
        iter = max(avail_iters);
    otherwise
        error( 'Wrong specificy of iter' );
end
