function custom_dir = nldetInternalTrainSnapshotDir( PARAM, is_deploy )

assert( ~PARAM.TrainSnapshot_External, ...
    'Internel error: External TrainSanpshot should not go through this function' );

if ~exist('is_deploy','var') || isempty(is_deploy)
    is_deploy = 0;
end

base_dir = sysStageSpecificDir( PARAM, 'Train', 0, 1, 1, 1 );

if is_deploy
    custom_dir = fullfile(base_dir,'latest');
else
    [availableIters, availableIters_str] = nldetAvailTrainSnapshotIters( base_dir );

    [c,p] = ismember( PARAM.TrainSnapshot_Iter, availableIters );

    if c
        custom_dir = fullfile( base_dir, sprintf( 'iter_%s', availableIters_str{p} ) );
    else
        custom_dir = [];
    end
end


