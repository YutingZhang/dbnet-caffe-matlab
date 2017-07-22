

t1 = tic_print( 'Search snapshot ( < %d ) : ',  PARAM.Train_StartIteration);
[ restore_dir, restore_iter ] = nldetNearestTrainSnapshot( ...
    PARAM.Train_StartIteration, STAGE_SPECIFIC_DIR );
toc_print(t1);
if isempty(restore_dir)
    fprintf( 'No snapshot is found, \n' );
    if PARAM.Train_Finetune
        % load finetune point
        finetune_point = SPECIFIC_DIRS.InitSnapshot;
        t1 = tic_print( 'Finetune from : %s\n', finetune_point );
        npl.load_pretrained( finetune_point );
        toc_print( t1 );
    else
        fprintf( 'Train from scratch (pretrained blocks were still loaded if possible).\n' );
    end
else
    % load snapshot
    t1 = tic_print( 'Load snapshot : %s\n It is at Iter %d : ', ...
        restore_dir, restore_iter );
    npl.load( restore_dir );
    toc_print(t1);
end