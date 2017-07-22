classdef IndexSampler < handle
    properties ( GetAccess= public, SetAccess = protected )
        rand_stream
        rand_stream_aux
        param
        chosenInds
        shuffledInds
        shuffledOrder
        numInds
        curOrder
        curEpoch
        
        use_unique_guard
        atomic_indexer = [];
    end
    methods
        function obj = IndexSampler( chosenIndexes, PARAM )
            if nargin<2, PARAM = struct(); end
            Pdef = struct();
            Pdef.shuffle         = false;
            Pdef.beginning_epoch = 0;
            Pdef.beginning_pos   = inf;
            Pdef.rand_seed       = 31841;
            Pdef.rand_seed_aux   = 'inherent';
            Pdef.limit           = inf; % maximum number of pool
            
            % path to a unique guard file. 
            % if non-empty, the index will be only go through once,
            % according to the guard file.
            % conflicting with shuffle
            Pdef.unique_guard    = [];
            
            PARAM = xmerge_struct( Pdef, PARAM );
            if strcmp( PARAM.rand_seed_aux, 'inherent' )
                PARAM.rand_seed_aux = PARAM.rand_seed;
            end
            obj.param = PARAM;
            if islogical( chosenIndexes )
                chosenInds0 = find( chosenIndexes );
            else
                chosenInds0 = chosenIndexes;
            end
            
            num_chosen = numel(chosenInds0);
            sample_limit = min(PARAM.limit, num_chosen);
            if sample_limit<num_chosen
                if sample_limit<=1
                    cidx1 = 1:sample_limit;
                else
                    cidx1 = round( 1:((num_chosen-1)/(sample_limit-1)):num_chosen );
                end
                obj.chosenInds = chosenInds0(cidx1);
            else
                obj.chosenInds = chosenInds0;
            end
            obj.chosenInds = vec(obj.chosenInds).';
            
            obj.shuffledOrder = 1:length(obj.chosenInds);
            obj.shuffledInds  = obj.chosenInds;
            obj.numInds  = length(obj.chosenInds);
            obj.curOrder = min( PARAM.beginning_pos, obj.numInds );
            obj.curEpoch = PARAM.beginning_epoch;
            obj.rand_stream = RandStream('mrg32k3a','seed',PARAM.rand_seed);
            obj.rand_stream_aux = RandStream('mrg32k3a','seed',PARAM.rand_seed_aux);
            
            obj.use_unique_guard = ~isempty(PARAM.unique_guard);
            if obj.use_unique_guard
                assert( ~obj.param.shuffle, 'shuffle is conficting with unique_guard' );
                obj.atomic_indexer = atomic_indexer( PARAM.unique_guard, obj.numInds );
            end
            
        end
        function sample_count = count(obj)
            sample_count = obj.numInds;
        end
        function epoch_n = epoch(obj)
            epoch_n = obj.curEpoch;
        end
        function iter_in_epoch = iter_in_epoch(obj)
            iter_in_epoch = obj.curOrder;
        end
        function cur_pos = pos(obj)
            cur_pos = obj.shuffledOrder(obj.curOrder);
        end
        function sample_order = next(obj)
            if obj.use_unique_guard
                if isempty(obj.curEpoch)
                    k = [];
                elseif obj.curEpoch==0
                    k = obj.atomic_indexer.start();
                else
                    k = obj.atomic_indexer.next();
                end
                if ~isempty(k)
                    if k<=0,
                        k = [];
                        obj.curEpoch = [];
                    end
                end
                obj.curOrder = k;
            else
                obj.curOrder = obj.curOrder + 1;
                if obj.curOrder > obj.numInds % start a new epoch
                    obj.curEpoch = obj.curEpoch + 1;
                    obj.curOrder = 1;
                    if obj.param.shuffle
                        obj.shuffledOrder(1:obj.numInds) = [];
                        while numel(obj.shuffledOrder)<obj.numInds*2
                            shuffle_order = obj.rand_stream.randperm(obj.numInds);
                            obj.shuffledOrder = [obj.shuffledOrder,shuffle_order];
                        end
                        obj.shuffledInds  = obj.chosenInds(obj.shuffledOrder);
                    end
                end
            end
            sample_order = obj.current();
        end
        function sample_order = current(obj)
            sample_order = obj.shuffledInds(obj.curOrder);
        end
        
        function sample_order = forecast(obj, k)
            b = obj.curOrder;
            if isempty(b), b = 0; end
            j = b+k;
            j = mod(j-1,numel(obj.shuffledInds))+1;
            sample_order = obj.shuffledInds(j);
        end
        
        % get a random index (it does not affect the current position)
        function [image_order, pos_in_subset] = random( obj, num_of_image )
            if nargin<2
                num_of_image = 1;
            end
            pos_in_subset = max(1,ceil(obj.rand_stream_aux.rand(num_of_image, 1)*obj.numInds));
            image_order = reshape( obj.chosenInds(pos_in_subset), num_of_image, 1);
        end
        
        % snapshot
        function snapshot( obj, snapshot_prefix )
            
            S = obj2struct(obj);
            shapshot_path = [ snapshot_prefix '.mat' ];
            save_from_struct( shapshot_path, S );
            
        end
        
        %
        function restore( obj, snapshot_prefix )
            obj.try_restore( snapshot_prefix, 1 );
        end
        %
        function try_restore( obj, snapshot_prefix, no_try )
            if nargin<3
                no_try = 0;
            end
            
            shapshot_path = [ snapshot_prefix '.mat' ];
            file_existed = boolean( exist(shapshot_path, 'file') );
            if no_try
                assert( file_existed, 'snapshot does not exist' );
            else
                return;
            end
            
            S = load( snapshot_prefix ); 
            F = fieldnames(S);
            for k = 1:length(F)
                if isprop( obj, F{k} )
                    ss  = substruct( '.', F{k} );
                    obj = subsasgn( obj, ss, ...
                        subsref(S,ss) );
                end
            end
            
        end
        
    end
    
    
end
