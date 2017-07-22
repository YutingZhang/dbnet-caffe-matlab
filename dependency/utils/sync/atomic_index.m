function [cur_task_index, more_info] = atomic_index( index_file_path, cmd, varargin )
% ATOMIC_INDEX is a utility for distributing jobs to different MATLAB instances
% 
% Usage: 
%   ... = atomic_index( index_file_path, cmd, ... )
%
% Get a task:
%   [cur_task_index, more_info] = atomic_index( index_file_path, 'inc'/'next', timeout, [, total_task_number] )
%     total_task_number must be provided for the first time
% The function is blocked if no task can be assigned while not all the tasks are done
%
% Reset the index file:
%   [cur_task_index, more_info] = atomic_index( index_file_path, 'reset', total_task_number )
%
% Start the index file:
%   [cur_task_index, more_info] = atomic_index( index_file_path, 'start', timeout, total_task_number, forced )
%  if forced = 0, then call INC/NEXT; otherwise, call reset
%
% Complete a task:
%   [cur_task_index, more_info] = atomic_index( index_file_path, 'done', task_id )
%  if success, cur_task_index = task_id; otherwise, 0;
%
% Keep a task alive (so that it is not timed out and taken by other workers):
%   [cur_task_index, more_info] = atomic_index( index_file_path, 'alive', task_id )
%  if success, cur_task_index = task_id; otherwise, 0;
%
% Get the status of the indexer:
%   [cur_task_index, more_info] = atomic_index( index_file_path, 'status' )
%  cur_task_index = 0
%

cmd = lower(cmd);

VARIN = varargin;

if strcmp(cmd,'start')
    assert( length(VARIN)==3, 'Exact three argument for START' );
    is_forced = VARIN{3};
    if is_forced
        cmd = 'reset';
        VARIN= VARIN(2);
    else
        cmd = 'inc';
        VARIN= VARIN(1:2);
    end
end

VAROUT = cell(1,7);
switch cmd
    case {'inc','next'}
        if length(VARIN)<1, error( 'At least one argument is need for INC/NEXT' ); end
        if length(VARIN)>2, error( 'At most two argument is need for INC/NEXT' ); end
        time_out = VARIN{1};
        if length(VARIN)>=2
            total_task_number = VARIN{2};
        else
            total_task_number = 0;
        end
        
        [VAROUT{:}] = atomic_index_switcher( index_file_path, cmd, total_task_number, time_out, 0 );
        [cur_task_index, more_info] = atomic_index_raw2formatted_output( VAROUT );
        
        while ~cur_task_index && more_info.complete<more_info.total
            pause( max(1,ceil(time_out/2)) );
            [VAROUT{:}] = atomic_index_switcher( index_file_path, cmd, total_task_number, time_out, 0 );
            [cur_task_index, more_info] = atomic_index_raw2formatted_output( VAROUT );
        end
        
    case 'reset'
        assert( length(VARIN)==1, 'Exact one argument for RESET' );
        total_task_number = VARIN{1};
        [VAROUT{:}] = atomic_index_switcher( index_file_path, 'reset', total_task_number, 0, 0 );
        [cur_task_index, more_info] = atomic_index_raw2formatted_output( VAROUT );
        
    case {'done','alive'}
        assert( length(VARIN)==1, 'Exact one argument for DONE/ALIVE' )
        task_id = VARIN{1};
        [VAROUT{:}] = atomic_index_switcher( index_file_path, cmd, 0, 0, task_id );
        [cur_task_index, more_info] = atomic_index_raw2formatted_output( VAROUT );
    case {'status'}
        assert( isempty(VARIN), 'No argument for DONE/ALIVE' )
        [VAROUT{:}] = atomic_index_switcher( index_file_path, cmd, 0, 0, 0 );
        [cur_task_index, more_info] = atomic_index_raw2formatted_output( VAROUT );
    otherwise
        error('Unrecognized command');
end

end

function [cur_task_index, more_info] = ...
    atomic_index_raw2formatted_output( VAROUT )

cur_task_index = VAROUT{1};
more_info = struct( ...
    'total',      VAROUT{2}, ...
    'next_task',  VAROUT{3}, ...
    'complete',   VAROUT{4}, ...
    'left_total', VAROUT{5}, ...
    'next_start_time', VAROUT{6}, ...
    'last_task',  VAROUT{7});

end
