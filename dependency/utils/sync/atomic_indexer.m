classdef atomic_indexer < handle
    
    properties (GetAccess=public,SetAccess=protected)
        index_file_path
        total_task_number
        
        reset_task_id = []
        reset_more_info = []
        reset_tic     = []
        
        predone_func = @(n) 0;
    end
    properties (Access=public)
        timeout
    end
    
    methods
        function obj = atomic_indexer( index_file_path, total_task_number, timeout )
            if nargin<3
                timeout = 30;
            end
            obj.index_file_path   = index_file_path;
            obj.total_task_number = total_task_number;
            obj.timeout = timeout;
        end
        function set_timeout( obj, timeout )
            obj.timeout = timeout;
        end
        function set_predone_func( obj, predone_func )
            if nargin<2 || isempty(predone_func)
                predone_func = @(n) 0;
            end
            obj.predone_func = predone_func;
        end
        function reset( obj )
            obj.reset_tic = tic;
            [cur_task_id, more_info] = atomic_index( ...
                obj.index_file_path, 'reset', obj.total_task_number );
            obj.reset_task_id   = cur_task_id;
            obj.reset_more_info = more_info;
        end
        function [cur_task_id, more_info] = inc( obj )
            [cur_task_id, more_info] = try_to_get_reset_task( obj );
            while obj.update_predone(cur_task_id)
                [cur_task_id, more_info] = atomic_index( ...
                    obj.index_file_path, 'inc', obj.timeout, obj.total_task_number );
            end
        end
        function [cur_task_id, more_info] = next( obj )
            [cur_task_id, more_info] = obj.inc();
        end
        function [cur_task_id, more_info] = start( obj, reset_at_beginning )
            if nargin>=2 && reset_at_beginning,
                obj.reset();
            end
            [cur_task_id, more_info] = obj.inc();
        end        
        function done( obj, task_id )
            [cur_task_id, more_info] = atomic_index( ...
                obj.index_file_path, 'done', task_id );
        end
        function keep_alive( obj, task_id )
            [cur_task_id, more_info] = atomic_index( ...
                obj.index_file_path, 'alive', task_id );
        end
        function [cur_task_id, more_info] = status( obj )
            [cur_task_id, more_info] = atomic_index( ...
                obj.index_file_path, 'status' );
        end
    end
    
    methods (Access=protected)
        function [cur_task_id, more_info] = try_to_get_reset_task( obj )
            cur_task_id = [];
            more_info   = [];
            if ~isempty(obj.reset_tic)
                if toc(obj.reset_tic)<obj.timeout
                    obj.keep_alive( obj.reset_task_id );
                    cur_task_id = obj.reset_task_id;
                    more_info   = obj.reset_more_info;
                end
                obj.reset_tic       = [];
                obj.reset_task_id   = [];
                obj.reset_more_info = [];
            end
        end
        function is_done = update_predone( obj, task_id )
            if isempty(task_id)
                is_done = 1;
                return;
            end
            is_done = obj.predone_func(task_id);
            if is_done, obj.done( task_id ); end
        end
    end
end
