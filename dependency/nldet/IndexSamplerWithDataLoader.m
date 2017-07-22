classdef IndexSamplerWithDataLoader < IndexSampler
    
    properties (GetAccess=public, SetAccess=protected)
        pre_loader_func
        loader_func
        prefetch_size
        
        cached_data_id = 0
        cached_data = []
        
        prefetched_idx
        prefetched_data
        prefetched_pos
        prefeteched_update_idx = []
        
        pending_prefetch = []
        
        tmp_data_idx
        tmp_pre_data = {}
        
    end
    
    methods
        function obj = IndexSamplerWithDataLoader( chosenIndexes, PARAM )
            Pdef = struct();
            Pdef.pre_loader_func = @(a) a;
            Pdef.loader_func  = @(a) [];
            Pdef.prefetch_size = 0;
            
            PARAM = xmerge_struct(Pdef,PARAM);
            
            obj@IndexSampler( chosenIndexes, PARAM );
            
            obj.pre_loader_func = PARAM.pre_loader_func;
            obj.loader_func     = PARAM.loader_func;
            obj.prefetch_size   = PARAM.prefetch_size;
            
            obj.reset_prefetched();
        end
        
        function set_loader_func( obj, loader_func )
            if nargin<2
                loader_func = @(a) [];
            end
            obj.loader_func = loader_func;
        end
        
        function set_pre_loader_func( obj, pre_loader_func )
            if nargin<2
                pre_loader_func = @(a) a;
            end
            obj.pre_loader_func = pre_loader_func;
        end
        
        function set_prefetch_size( obj, new_prefetch_size )
            obj.prefetch_size = new_prefetch_size;
            obj.reset_prefetched();
        end
        
        function reset_prefetched( obj )
            obj.prefetched_idx  = zeros( 1, obj.prefetch_size );
            obj.prefetched_data = cell( 1, obj.prefetch_size );
            obj.prefetched_pos  = 1;
            obj.prefeteched_update_idx = [];
        end
        
        function sample_order = next( obj )
            sample_order = next@IndexSampler( obj );
            obj.prefetched_pos = mod(obj.prefetched_pos,obj.prefetch_size) + 1;
        end
        
        function prefetch( obj )
            
            if obj.prefetch_size<1, return; end
            
            if ~isempty(obj.prefeteched_update_idx)
                obj.join();
            end
            
            forecasted_idx = obj.forecast(1:obj.prefetch_size);
            pidx =[obj.prefetched_pos:obj.prefetch_size,1:(obj.prefetched_pos-1)];
            update_raw_idxb = (obj.prefetched_idx(pidx) ~= forecasted_idx);
            
            if ~any(update_raw_idxb), return; end
            
            update_idx  = pidx(update_raw_idxb);
            
            obj.prefeteched_update_idx = update_idx;
            obj.tmp_data_idx = forecasted_idx(update_raw_idxb);
            obj.tmp_pre_data = arrayfun( obj.pre_loader_func, ...
                obj.tmp_data_idx , ...
                'UniformOutput', 0);
            
            obj.pending_prefetch = try_parfeval( 1, ...
                @cellfun, 1, obj.loader_func, obj.tmp_pre_data, ...
                'UniformOutput', 0 );
            
        end
        
        function join( obj )
            
            if obj.prefetch_size<1, return; end
            
            if isempty(obj.prefeteched_update_idx) || ...
                    isempty(obj.pending_prefetch)
                return;
            end
            
            pd = try_fetchOutputs(obj.pending_prefetch);
            obj.prefetched_data( obj.prefeteched_update_idx ) = pd;
            
            obj.prefetched_idx( obj.prefeteched_update_idx ) = obj.tmp_data_idx;
            
            obj.prefeteched_update_idx = [];
            
        end
        
        function D = get_data( obj )
                        
            cur_id = obj.current();
            
            if obj.cached_data_id == cur_id
                D = obj.cached_data;
                return;
            end
            
            use_prefetch = 0;
            if obj.prefetch_size>0
                matched_prefetch = ( obj.prefetched_idx == cur_id );
                if any(matched_prefetch)
                    D = obj.prefetched_data{find(matched_prefetch,1)};
                    use_prefetch = 1;
                end
            end
            
            if ~use_prefetch
                pre_data = obj.pre_loader_func(cur_id);
                D = obj.loader_func(pre_data);
            end
            
            obj.cached_data_id = cur_id;
            obj.cached_data    = D;
            
        end
        
    end
    
end
