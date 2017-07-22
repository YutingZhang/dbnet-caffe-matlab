
npl_snapshot = @() nldetPipelineSnapshot( npl, STAGE_SPECIFIC_DIR );
npl_step = @() npl.step;
while npl.train_self_iter < PARAM.Train_MaxIteration
    
    if PARAM.Train_SnapshotOnInterruption
        run_with_interruption_handler( npl_step, npl_snapshot );
    else
        npl_step();
    end
    
    % do snapshot if needed
    do_snapshot = 0;
    if ~isempty(PARAM.Train_SnapshotFrequency) && ...
            ~mod( npl.train_self_iter, ...
            PARAM.Train_SnapshotFrequency )
        do_snapshot = 1;
    end
    if ~isempty(PARAM.Train_SnapshotAt) && ...
            ismember( npl.train_self_iter, ...
            PARAM.Train_SnapshotAt )
        do_snapshot = 1;
    end
    
    if do_snapshot
        npl_snapshot();
    end
    
end

