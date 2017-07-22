classdef onMutableCleanup < handle

    properties (GetAccess = public, SetAccess = protected)
        task
    end
    
    methods
        
        function obj = onMutableCleanup( task )
            if nargin<1
                task = [];
            end
            obj.task = task;
        end
        
        function delete( obj )
            if ~isempty( obj.task )
                feval( obj.task );
            end
        end
        
        function reset( obj, task )
            if nargin<2
                task = [];
            end
            obj.task = task;
        end
        
    end

end
