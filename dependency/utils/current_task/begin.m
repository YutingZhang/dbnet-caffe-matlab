function begin

global cur_task_stage_state__

if isempty( cur_task_stage_state__ ) || ~cur_task_stage_state__
    cur_task_stage_state__ = 1;
else
    current_task_exception_handler( 'begin: unpaired begin/over' );
end

end
