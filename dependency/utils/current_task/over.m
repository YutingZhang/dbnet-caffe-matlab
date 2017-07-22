function over

global cur_task_stage_state__

if isempty( cur_task_stage_state__ ) || ~cur_task_stage_state__
    current_task_exception_handler( 'over: unpaired begin/over' );
    % still go on
end
cur_task_stage_state__ = 0;

end
