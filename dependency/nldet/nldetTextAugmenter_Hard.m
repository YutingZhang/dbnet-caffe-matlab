classdef nldetTextAugmenter_Hard < handle
    
    properties ( Access = private )
        hard_text
        text_pool_source
        text_pool_phrase
        rand_stream
        
        use_source2phrase_func
        
        param
    end
    
    
    methods
        function obj = nldetTextAugmenter_Hard( image_num, PARAM )
            
            if ~exist('PARAM','var') || isempty(PARAM)
                PARAM = struct();
            end
            
            Pdef = struct();
            Pdef.image_chosen  = 5;  % number of chosen neg samples (image wide) 
            Pdef.region_chosen = 0;  % number of chosen neg samples (per region)
            Pdef.image_cached  = 10; % number of cached neg samples (per region)
            Pdef.region_cached = 0;  % number of cached neg samples (per region)
            Pdef.hard_threshold = 0.5; % this is the default for logistic regression
            
            Pdef.rand_seed = 21749;
            Pdef.source2phrase_func = [];
            
            PARAM = obj.process_param(Pdef, PARAM);
            
            obj.text_pool_source = zeros(0,1);
            obj.text_pool_phrase = cell(0,1);

            obj.hard_text = repmat( struct( 'image', {{}}, 'regions', {{}} ), ...
                image_num, 1 );
            obj.rand_stream   = RandStream('mrg32k3a','seed',PARAM.rand_seed);
            
            obj.param = PARAM;
            
            obj.use_source2phrase_func = ~isempty( ...
                obj.param.source2phrase_func );
            
        end
    end

    methods ( Static )
        function T = empty_text()
            T = struct('phrase',{},'source',{});
        end
        function PARAM = process_param( Pdef, PARAM )
            PARAM = xmerge_struct( Pdef, PARAM );
            if strcmp( PARAM.region_chosen, 'inherent' )
                PARAM.region_chosen = PARAM.image_chosen;
            end
            if strcmp( PARAM.region_cached, 'inherent' )
                PARAM.region_cached = PARAM.image_cached;
            end
            PARAM.image_cached  = max(PARAM.image_cached,  PARAM.image_chosen);
            PARAM.region_cached = max(PARAM.region_cached, PARAM.region_chosen);
            
            assert( PARAM.region_cached==0 && PARAM.region_chosen==0, ...
                'region-wise hard mining has not been implemented.' );
        end
    end
    
    methods ( Access = protected )

        function initial_image( obj, image_position )
            p = image_position;
            if isempty(obj.hard_text(p).image) && iscell(obj.hard_text(p).image)
                % intialization
                obj.hard_text(p).image = [];
%                 empty_hard_list = struct( 'text_id', {}, 'difficulty', {} );
%                 obj.hard_text(p).image = empty_hard_list;
            end
        end
        function initial_region( obj, image_position, region_order )
            p = image_position;
            k = region_order;
            k0 = numel(obj.hard_text(p).regions);
            if k0<k
                obj.hard_text(p).regions((k0+1):k) = {[]};
%                 empty_hard_list = struct( 'text_id', {}, 'difficulty', {} );
%                 obj.hard_text(p).regions((k0+1):k) = {empty_hard_list};
            end
        end
        function t = text_from_ids( obj, text_ids )
            if obj.use_source2phrase_func
                phrases = obj.param.source2phrase_func(text_ids);
                t = struct( 'phrase', vec(phrases), 'source', num2cell(vec(text_ids)) );
            else
                [in_pool, pos_pool] = ismember( text_ids, obj.text_pool_source );
                assert( all(in_pool), 'id not in pool' );
                t = struct( 'phrase', vec(obj.text_pool_phrase(pos_pool)), ...
                    'source', num2cell(vec(obj.text_pool_source(pos_pool))) );
            end
        end
        function text_ids = add2text_pool( obj, t )
            src_ids = vec([t.source]);
            if obj.use_source2phrase_func
                text_ids = src_ids;
            else
                [in_pool, pos_pool] = ismember( src_ids, obj.text_pool_source );
                n = numel(obj.text_pool_source);
                obj.text_pool_source = [obj.text_pool_source;
                    vec([t(~in_pool).source])];
                obj.text_pool_phrase = [obj.text_pool_phrase;
                    vec([t(~in_pool).phrase])];
                pos_pool(~in_pool) = n+(1:sum(~in_pool));
                text_ids = pos_pool;
            end
        end
    end
    
    methods
        
        function set_param( obj, PARAM )
            PARAM1 = obj.process_param(obj.param,PARAM);
            obj.param = PARAM1;
        end
        
        function T = image_neg( obj, im )
            
            % im.order
            % im.position
                        
            num_chosen_text = obj.param.image_chosen;
            if num_chosen_text>0
                p = im.position;           
                initial_image( obj, p );

                htid = [obj.hard_text(p).image].';
                if length(htid)>num_chosen_text
                    htid = htid( obj.rand_stream.randperm(length(htid), ...
                        min(length(htid),num_chosen_text)) );
                end
                T = obj.text_from_ids( htid );
            else
                T = obj.empty_text();
            end

            
        end

        % do not support regional hard mining for now
        %{
        function T = region_neg( obj, im, rgn )
            
            % im.order
            % im.position
            
            % rgn.order
            % rgn.text
            
            num_chosen_text = obj.param.region_chosen;
            
            if num_chosen_text>0
                p = im.position;
                k = rgn.order;
                initial_region( obj, p, k );            
                
                htid = [obj.hard_text(p).regions{k}].';
                if length(htid)>num_chosen_text
                    htid = htid( obj.rand_stream.randperm(length(htid), ...
                        min(length(htid),num_chosen_text)) );
                end
                T = obj.text_from_ids( htid );  
            else
                T = obj.empty_text();
            end
            
        end
        %}
        
        function feedback( obj, im, pair_scores, D )
            
            [gt_in_pool, gt_pos_pool] = ismember( ...
                [D.img.regions.gt_source],[D.text_pool.source]);
            assert( all(gt_in_pool), 'gt_source does not exist in pool' );
            
            p = im.position;
            
            R = codebook_cols( pair_scores, {'region_id','text_id','label','score'} );
            R( R(:,3)>0, : ) = []; % only considier negatives
            R( ismember( R(:,2), gt_pos_pool ), : ) = []; % do not take annotated text as hard
            hard_neg_idxb = ( R(:,4) > obj.param.hard_threshold );
            R( ~hard_neg_idxb, : ) = [];
            
            % find image hard negative
            if ~isempty(R)
                initial_image( obj, p );

                [u_text_imIds,~,text_imId_order2results] = unique( R(:,2) );
                u_text_scores = accumarray( text_imId_order2results, ...
                    R(:,4), [], @max );
                [u_text_scores, sidx] = sort( u_text_scores, 'descend' );
                u_text_imIds = u_text_imIds(sidx);
                
                cur_active_idxb = ismember( obj.hard_text(p).image, u_text_imIds );
                stalled_num  = numel( obj.hard_text(p).image ) - sum(cur_active_idxb);
                refresh_num  = obj.param.image_cached - stalled_num;
                new_hard_num = numel(u_text_imIds);
                chosen_hard_num = max(0, min( refresh_num, new_hard_num ) );
                % remark: use max(0,___) in case the cache shrinks
                
                u_hard_text_ids = obj.add2text_pool( ...
                    D.text_pool( u_text_imIds(1:chosen_hard_num) ) );
                obj.hard_text(p).image = [ vec(u_hard_text_ids);
                    obj.hard_text(p).image(~cur_active_idxb)];
            end
            
            
            % add ids to the cache
            
            % find region hard negative *****************************
            % too complicated, has not implemented right now
            
        end
        
        % function to snapshot
        function snapshot( obj, output_prefix )
            fn = [output_prefix '.v7.mat'];
            S = scalar_struct( ...
                'hard_text',         obj.hard_text, ...
                'text_pool_source',  obj.text_pool_source, ...
                'text_pool_phrase', obj.text_pool_phrase, ...
                'rand_stream',       obj.rand_stream, ...
                'use_source2phrase_func', obj.use_source2phrase_func );
            mkpdir_p( output_prefix );
            save7( fn, S );
        end
        
        function try_restore( obj, input_prefix, no_try )
            
            if nargin<3
                no_try = 0;
            end
            
            fn = [input_prefix '.v7.mat'];
            file_existed = boolean(exist(fn, 'file'));
            if no_try
                assert( file_existed, 'File does not exist' );
            else
                return;
            end
            S = load7( fn );
            if boolean(obj.use_source2phrase_func) ~= ...
                    boolean( S.use_source2phrase_func )
                if no_try, 
                    error( 'conflict in use_source2phrase_func.' );
                else
                    warning( 'conflict in use_source2phrase_func. do not restore' );
                end
            end
            
            obj.hard_text         = S.hard_text;
            obj.text_pool_source  = S.text_pool_source;
            obj.text_pool_phrase  = S.text_pool_phrase;
            obj.rand_stream       = S.rand_stream;

        end
        
        function restore( obj, input_prefix )
            
            obj.try_restore( input_prefix, 1 );
            
        end

        
        
    end
    
end
