classdef nldetBatchSampler < handle
    properties ( GetAccess = public, SetAccess = protected )
        
        param
        batch_size
        image_sampler
        gt_loader
        D % current data
        
        % text augmenter
        taug_list
        
        I % tmp images
        T % tmp text
        
        score_remapping_func
        
        iters_in_epoch
        
        processing_level;
        
        has_text_comp
        
    end
    
    properties (Constant)
        
        PLEVEL_GET          = 0
        PLEVEL_ADD_BOX      = 1
        PLEVEL_PROP_TEXT    = 2
        PLEVEL_FEEDBACK     = 3
        PLEVEL_DONE         = 10
        PLEVEL_START        = 1000
        PLEVEL_INVALID      = inf;
        
    end
        
    methods ( Access = public )
        function obj = nldetBatchSampler( batch_size, ...
                image_sampler, gt_loader, PARAM )
            
            if ~exist('PARAM','var') || isempty(PARAM)
                PARAM = struct();
            end
            if ~exist('gt_loader','var')
                gt_loader = [];
            end
            Pdef = struct();
            
            % parameters
            Pdef.random_at_gt = 0;
            
            Pdef.rand_seed = 80432;
            
            Pdef.gt_nearby_threshold = 0.3; % threshold to determine whether a box is close to a gt
            
            % determine if take gt text as neg text for other boxes in the same image 
            Pdef.same_image_gt_as_neg = 1;
            
            % method to handling conflicting titles
            %  'scoring':   for testing, so that all pairs are tested, and
            %                ambiguous regions can be avoided by knowing
            %                the score.
            % % 'avoiding': for training, conflicting titles are all
            %                removed to avoid ambiguity 
            Pdef.conflicting_policy = 'avoiding';
            
            % a string to specific the remapping function (without @, must be defined in mfile)
            % it can output NaN which removes the boxes
            Pdef.score_remapping_func_str = [];
            Pdef.score_remapping_func_args = {};
            
            Pdef.text_compatibility_threshold = 1;
            
            PARAM = xmerge_struct( Pdef, PARAM );
            
            assert( ismember( PARAM.conflicting_policy, {'scoring','avoiding'} ), ...
                'wrong conflicting ' ); 
            
            obj.batch_size    = batch_size;
            obj.image_sampler = image_sampler;
            obj.gt_loader     = gt_loader;
            obj.D             = []; % current data
            obj.param         = PARAM;
            
            % initialize rand stream
            % obj.rand_stream = RandStream('mrg32k3a','seed',PARAM.rand_seed);
            
            if isempty(obj.param.score_remapping_func_str)
                obj.score_remapping_func = [];
            else
                score_remapping_func = eval( ['@' obj.param.score_remapping_func_str] );
                obj.score_remapping_func = @(s) score_remapping_func(s, ...
                    obj.param.score_remapping_func_args{:});
            end
            
            % has text compitable
            obj.has_text_comp = has_method_x( obj.gt_loader, 'text_comp' );
            
            % initialize addons
            obj.taug_list = struct( 'id_str',{}, ...
                'image', {}, 'image_neg', {}, 'image_pos', {},...
                'region', {}, 'region_neg',{}, 'region_pos',{}, ...
                'feedback', {}, 'taug',{});
            
            % set processing level
            obj.processing_level = obj.PLEVEL_START;

        end
        
        function add_text_augmenter( obj, id_str, taug )
            
            % check input
            assert( ischar(id_str) && isrow(id_str), 'id_str must be string' );
            assert( ~ismember(id_str, {obj.taug_list.id_str}), ...
                sprintf( 'text_augmenter ''%s'' has been registered', id_str ) );
            
            % add the text augmenter
            ntaug = struct();
            ntaug.id_str = id_str;
            ntaug.image      = ismethod( taug, 'image' );
            ntaug.image_neg  = ismethod( taug, 'image_neg' );
            ntaug.image_pos  = ismethod( taug, 'image_pos' );
            ntaug.region     = ismethod( taug, 'region' );
            ntaug.region_neg = ismethod( taug, 'region_neg' );
            ntaug.region_pos = ismethod( taug, 'region_pos' );
            ntaug.feedback   = ismethod( taug, 'feedback' );
            ntaug.taug       = taug;
            assert( ntaug.image || ntaug.image_neg || ntaug.image_pos || ...
                ntaug.region || ntaug.region_neg || ntaug.region_pos, ...
                'No augmenter function is found in the augmenter' );
            obj.taug_list(end+1) = ntaug;
                        
        end

        function C = text_comp( obj, query_ids, gallary_ids )
            
            if obj.has_text_comp
                C = obj.gt_loader.text_comp( query_ids, gallary_ids );
            else
                C = bsxfun( @eq, vec(query_ids), vec(gallary_ids).' );
            end
            
        end
        
        function remove_text_augmenter( obj, id_str )
            
            % check input
            assert( ischar(id_str) && isrow(id_str), 'id_str must be string' );
            [in_list,list_pos] = ismember(id_str, {obj.taug_list.id_str});
            % assert( in_list, sprintf( 'text_augmenter ''%s'' has NOT been registered', id_str ) );
            % remove the text augmenter
            if in_list
                obj.taug_list(list_pos) = [];
            end
            
        end
        
        function taug = get_text_augmenter( obj, id_str )
            
            % check input
            assert( ischar(id_str) && isrow(id_str), 'id_str must be string' );
            [in_list,list_pos] = ismember(id_str, {obj.taug_list.id_str});
            % assert( in_list, sprintf( 'text_augmenter ''%s'' has NOT been registered', id_str ) );
            % return the text augmenter
            if in_list
                taug = obj.taug_list(list_pos).taug;
            else
                taug = [];
            end
            
        end

        %{
        function save_cache( obj, fn )
            assert( fn, 'fn should be a string of the cache path' );
            S = struct( 'text_pool', {obj.text_pool}, 'hard_text', {obj.hard_text} );
            save7( fn, S );
        end
        
        function load_cache( obj, fn )
            assert( fn, 'fn should be a string of the cache path' );
            assert( exist(fn,'file')~=0, 'the cache does not exist' );                
            S = obj.load7( PARAM.from_cache );
            assert( numel(S.hard_text)==numel(obj.hard_text), ...
                'cache length for hard_text does not match' );
            obj.text_pool = S.text_pool;
            obj.hard_text = S.hard_text;
        end
        %}
        
        function epoch_count = epoch( obj )
            epoch_count = obj.image_sampler.epoch;
        end
        
        function cur_data = current( obj )
            cur_data = obj.D;
        end
        
        function set_random_choser( obj, phrase_number )
            if nargin < 1
                obj.remove_text_augmenter( '_Random' );
            else
                random_taug = obj.get_text_augmenter( '_Random' );
                if isempty(random_taug)
                    if phrase_number > 0
                        random_taug = nldetTextAugmenter_Random( ...
                            obj.image_sampler, obj.gt_loader, ...
                            phrase_number, obj.param.rand_seed * 11);
                        obj.add_text_augmenter( '_Random', random_taug );
                    end
                else
                    random_taug.set_phrase_number( 0 );
                end
            end
        end
        
        function set_hard_miner( obj, param )
            if nargin < 1
                obj.remove_text_augmenter( '_Hard' );
            else
                hard_taug = obj.get_text_augmenter( '_Hard' );
                if isempty(hard_taug)
                    Pdef  = struct('rand_seed', obj.param.rand_seed*23);
                    param = xmerge_struct( Pdef, param );
                    if has_method_x( obj.gt_loader, 'phrase' )
                        param.source2phrase_func = ...
                            @(varargin) obj.gt_loader.phrase(varargin{:});
                    end
                    hard_taug = nldetTextAugmenter_Hard( ...
                        obj.image_sampler.count(), param);
                    obj.add_text_augmenter( '_Hard', hard_taug );
                else
                    hard_taug.set_param( param );
                end
            end
        end
        
        function c = text_compatible( obj, target_texts, gallary_texts )
            
            if iscell( gallary_texts )
                gallary_each_num = cellfun(@numel, gallary_texts);
                gallary_texts = cat(1,gallary_texts{:});
            else
                gallary_each_num = [];
            end
            if isempty(gallary_texts)
                c = false( numel(target_texts), 0 );
                return;
            end
            
            target_id  = cat(1,target_texts.source);
            gallery_id = cat(2,gallary_texts.source);
            
            c = obj.text_comp( target_id, gallery_id );
            if isnumeric(c)
                c = (c>=obj.param.text_compatibility_threshold);
            end
            
            if ~isempty(gallary_each_num) && ~all(gallary_each_num==1)
                grpId = RunLength( 1:numel(gallary_each_num), vec(gallary_each_num).' );
                U = zeros( [size(c), 2], 'uint64' );
                U(:,:,1) = repmat(uint64(1:numel(target_id)).',1,numel(gallary_id));
                U(:,:,2) = repmat(uint64(grpId),numel(target_id),1);
                U = reshape(U,numel(U)/2,2);
                c = accumarray( U, vec(c), [numel(target_id),numel(gallary_each_num)], @any );
            end
            
        end
        
    end
    
    methods ( Access = protected )
        
        function fill_single( obj, j )
            
            % load images
            cur_image_order   = obj.image_sampler.next();
            obj.I(j).order    = cur_image_order;   % order in the dataset
            obj.I(j).iter_in_epoch = obj.image_sampler.iter_in_epoch();
            if isempty(cur_image_order)
                return;
            end
            obj.I(j).position = obj.image_sampler.pos();    % position in the subset
            obj.I(j).meta     = obj.image_sampler.meta();
            [obj.I(j).im, obj.I(j).im_path] = obj.image_sampler.image();
            
            % get gt annotations
            empty_text_pool = struct('phrase',cell(0,1),'source',cell(0,1));
            is_empty_batch = 0;
            if isempty( obj.gt_loader )
                is_empty_batch = 1;
            else
                A = obj.gt_loader.get(obj.I(j).order);
                is_empty_batch = isempty(A);
            end
            if is_empty_batch
                % obj.I(j).boxes    = zeros(0,4);
                obj.I(j).regions  = reshape( struct('box',{},'text_id',{}, 'label', {}, ...
                    'box_rank', {}, 'is_gt', {}, 'gt_source', {}, 'is_added', {}, 'nearby_gt', {}), 0, 1 );
                obj.I(j).text_id = zeros(0,1);
                obj.I(j).label   = zeros(0,1);
                obj.I(j).text_pool = empty_text_pool;
                return; % if no gt loader provided, no futher annotation proposal is needed
            end
            
            A = reshape(A,numel(A),1);
            try
                the_text_pool = cat(1,empty_text_pool,A.text);
            catch
                the_text_pool = cat_struct(1,empty_text_pool,A.text);
            end
            obj.I(j).text_pool  = the_text_pool;
            numText = arrayfun(@(a) numel(a.text), A);
            pos_j   = arrayfun(@(a,b) (b-a)+(1:a).', numText, cumsum(numText), ...
                'UniformOutput', 0 );
            gt_source_j = arrayfun(@(a) [a.text.source], A, 'UniformOutput', 0); % source id of the gt text
            label_j = arrayfun(@(a) ones(a,1), numText, 'UniformOutput', 0 );
            obj.I(j).regions = struct( 'box', {A.box}.', ...
                'text_id', pos_j, 'label', label_j, ...
                'box_rank', 0, ...
                'is_gt', true, 'gt_source', gt_source_j , ...
                'is_added', false, 'nearby_gt', [] );
            % remark: labels (1 for pos, 0 for neg)
            obj.I(j).text_id = zeros(0,1);
            obj.I(j).label   = zeros(0,1);
            
            % random regions for each gt
            if obj.param.random_at_gt
                error('random regions has not been implemented right now');
            end
                        
            % run text augmenter
            augT_im = struct('order', obj.I(j).order, 'position', obj.I(j).position);
            for r = 1:numel(obj.taug_list)
                ts = obj.taug_list(r);
                if ts.image || ts.image_neg || ts.image_pos
                    augT = []; augL= [];
                    if ts.image
                        [augT1, augL1] = ts.taug.image(augT_im);
                        assert( numel(augL1) == numel(augT1), ...
                            'L (labels indication) should be the same size as T (phrases)' );
                        augT = [augT; vec(augT1)];
                        augL = [augL; vec(augL1)];
                    end
                    if ts.image_pos
                        augT1 = ts.taug.image_pos(augT_im);
                        augT = [augT; vec(augT1)];
                        augL = [augL; ones(numel(augT1),1)];
                    end
                    if ts.image_neg
                        augT1 = ts.taug.image_neg(augT_im);
                        augT = [augT; vec(augT1)];
                        augL = [augL; zeros(numel(augT1),1)];
                    end
                    cnText = numel( obj.I(j).text_pool );
                    obj.I(j).text_pool = [obj.I(j).text_pool; augT];
                    obj.I(j).text_id = [obj.I(j).text_id ; cnText+(1:length(augT)).'];
                    obj.I(j).label   = [obj.I(j).label   ; vec(augL)];
                end
                if ts.region || ts.region_neg || ts.region_pos
                    for k = numel(A):-1:1 % use reverse order for the efficiency of hard mining augmenter
                        augT_rgn = struct( 'order', {k}, 'text', {A(k).text} );
                        augT = []; augL= [];
                        if ts.region
                            [augT1, augL1] = ts.taug.region(augT_im, augT_rgn);
                            assert( numel(augL1) == numel(augT1), ...
                                'L (labels indication) should be the same size as T (phrases)' );
                            augT = [augT; vec(augT1)];
                            augL = [augL; vec(augL1)];
                        end
                        if ts.region_pos
                            augT1 = ts.taug.region_pos(augT_im, augT_rgn);
                            augT = [augT; vec(augT1)];
                            augL = [augL; ones(numel(augT1),1)];
                        end
                        if ts.region_neg
                            augT1 = ts.taug.region_neg(augT_im, augT_rgn);
                            augT = [augT; vec(augT1)];
                            augL = [augL; zeros(numel(augT1),1)];
                        end
                        cnText = numel( obj.I(j).text_pool );
                        obj.I(j).text_pool = [obj.I(j).text_pool; augT];
                        obj.I(j).regions(k).text_id = [ ...
                            obj.I(j).regions(k).text_id ; cnText+(1:length(augT)).'];
                        obj.I(j).regions(k).label = [ ...
                            obj.I(j).regions(k).label ; vec(augL)];
                    end
                end
            end
            
        end
        
        function fuse_batch( obj )
            % clean empty batch
            obj.I( arrayfun( @(a) isempty(a.order), obj.I ) ) = [];
            if isempty(obj.I),
                return;
            end
            % accumulate text
            numText = arrayfun(@(a) numel(a.text_pool), obj.I);
            obj.T = cat(1,obj.I.text_pool);
            [~, uniqTextIds, newTextIds] = unique({obj.T.phrase}.');
            obj.T = obj.T(uniqTextIds);
            newTextIds = mat2cell(newTextIds,numText,1);
            for j = 1:numel(obj.I)
                newTextIds_j = newTextIds{j};
                obj.I(j).regions = ...
                    nldetMapAndUniqueTitle( obj.I(j).regions, newTextIds_j );
                obj.I(j) = nldetMapAndUniqueTitle( ...
                    obj.I(j), newTextIds_j );
                for i = 1:numel(obj.I(j).regions)
                    obj.I(j).regions(i).gt_source = deal( ...
                        obj.T(obj.I(j).regions(i).text_id).source );
                end
            end
            obj.I = rmfield(obj.I,'text_pool');

        end
        
        % propagate text annotations on single images
        function propagate_text_single( obj, img_id )

            if isempty(obj.D), return; end
            
            tp = obj.D.text_pool;
            im = obj.D.img(img_id);

            % find gt regions and nearby gt
            boxes = double(cat(1, im.regions.box));
            gt_region_idx = find( cat(2,im.regions.is_gt) );
            ov = zeros( size(boxes,1), numel(gt_region_idx) );
            for j = 1:numel(gt_region_idx)
                k = gt_region_idx(j);
                ov(:,j) = PascalOverlap(boxes(k,:), boxes);
            end
            if obj.param.gt_nearby_threshold>0
                is_nearby = ( ov >= obj.param.gt_nearby_threshold );
            else
                is_nearby = ( ov > 0 );
            end
            for k = 1:size(boxes,1)
                nearby_gt_k = gt_region_idx( is_nearby(k,:) );
                im.regions(k).nearby_gt = nearby_gt_k; 
                % remark: old results (should be no old result) are always rewritten
            end
            
            % image negatives
            neg_text_ids = cell(numel(gt_region_idx)+1,1);
            img_neg_ids = (im.label==0);
            neg_text_ids{end} = cat(1, im.text_id(img_neg_ids) );
            im.text_id(~img_neg_ids) = []; % remove titles from image level, if they can be tranferred to bboxes
            im.label(~img_neg_ids) = [];
            
            % negative from GT positives (negative for others)
            % gt_region_idx = find( cat(2,im.regions.is_gt) ); % compute it already
            for j = 1:numel(gt_region_idx)
                k = gt_region_idx(j);
                % ** only use definite positive (not sure if it is necessary, ...
                %   or use label>0.5 a soft threshold for ambiguous labels)
                neg_text_ids{j} = im.regions(k).text_id(im.regions(k).label==1); 
            end
            gt_text_ids  = neg_text_ids(1:(end-1)); % support multiple GT text per region
            if obj.param.same_image_gt_as_neg
                neg_text_ids = cat(1,neg_text_ids{:});
            else
                neg_text_ids = neg_text_ids{end};
            end
            neg_text_ids = unique( neg_text_ids, 'stable' );
            
            % propagate positive text to boxes near gt
            
            gt_regions   = im.regions(gt_region_idx);
            text_ids_from_gt = cat(1, gt_regions.text_id );
            labels_from_gt   = cat(1, gt_regions.label );

            expanded_gt_title_id = RunLength((1:numel(gt_regions)).', ...
                cellfun(@numel,{gt_regions.text_id}.') );

            ov_x = ov(:,expanded_gt_title_id).';
            is_nearby_x = is_nearby(:,expanded_gt_title_id).';

            for k = find( any(is_nearby,2).' )
                % it should propagate of all titles
                % (including both positive and negative)
                chosen_idx = is_nearby_x(:,k);
                ov_k = ov_x(chosen_idx,k);
                prop_text_ids = text_ids_from_gt(chosen_idx);
                prop_labels   = labels_from_gt(chosen_idx) .* ov_k;

                im.regions(k).text_id = ...
                    [im.regions(k).text_id; prop_text_ids];
                im.regions(k).label   = ...
                    [im.regions(k).label; prop_labels];
            end
            im.regions = nldetMapAndUniqueTitle( im.regions, [], 'max' );

            
            % propagate image-level negatives (including gt from other regions) to bounding boxes
            % -- clean up negative annotations against gt
            
            % Remark: do not propagate positive annotation
            %  1) figure out all negative title (txtid-label) to propagate
            %  2) test all titles against each GT to figure out conflict regions
            %  3) propagate to other regions
            
            % determine whether the negative titles are against the GTs
            gt_text = cell(numel(gt_region_idx),1);
            for j = 1:numel(gt_region_idx)
                gt_text{j} = tp( gt_text_ids{j} );
            end
            
            %gt_conflict_idxb = false(numel(neg_text_ids),numel(gt_region_idx));
            gt_conflict_score = zeros(numel(neg_text_ids),numel(gt_region_idx));
            gt_conflict_score(:,:) = obj.text_compatible(tp(neg_text_ids),gt_text);
            
%             for i = 1:numel(neg_text_ids)
%                 target_text = tp(neg_text_ids(i));
%                 for j = 1:numel(gt_region_idx)
%                     tc = obj.text_compatible(target_text,gt_text{j});
%                     % any compatibility means a conflict
%                     %gt_conflict_idxb(i,j) = any(tc(:));
%                     gt_conflict_score(i,j) = max(double(tc(:)));
%                 end
%             end
            
            % propagate negative text
            switch obj.param.conflicting_policy
                case 'avoiding'
                    gt_conflict_idxb = boolean(gt_conflict_score>0);
                    % is_nearby:           [ size(boxes,1), numel(gt_region_idx),                   1 ]
                    % gt_conflict_idxb_rs: [             1, numel(gt_region_idx), numel(neg_text_ids) ]
                    gt_conflict_idxb_rs = permute(gt_conflict_idxb,[3,2,1]);
                    % box_conflict_idxb:   [ size(boxes,1), numel(gt_region_idx), numel(neg_text_ids)]
                    box_conflict_idxb = bsxfun( @and, is_nearby, gt_conflict_idxb_rs );
                    % nonconflict_idxb:    [ size(boxes,1), numel(neg_text_ids)]
                    nonconflict_idxb = ~any( box_conflict_idxb, 2 );
                    nonconflict_idxb = reshape( nonconflict_idxb, size(nonconflict_idxb,1), size(nonconflict_idxb,3) );
                    % neg_text_inset_idx should be the index of neg_text_ids
                    [neg_text_inset_idx,~] = ind2sub( size(nonconflict_idxb.'), find(nonconflict_idxb.') );
                    box_num_titles   = sum(nonconflict_idxb,2);
                    flatten_ntii = neg_text_ids(neg_text_inset_idx.');
                    flatten_ntii = mat2cell(flatten_ntii,box_num_titles,1);
                    for k = 1:size(boxes,1)
                        im.regions(k).text_id = [im.regions(k).text_id; flatten_ntii{k}];
                        im.regions(k).label   = [im.regions(k).label; zeros(size(flatten_ntii{k}))];
                    end
                    % merge non-unique title (txtid-label)
                    %   make label as negative as possible to avoid
                    %   training with false negative
                    im.regions = nldetMapAndUniqueTitle( im.regions, [], 'min' );
                    im         = nldetMapAndUniqueTitle( im, [], 'min' );
                case 'scoring'
                    gt_conflict_score_rs = permute(gt_conflict_score,[3,2,1]);  
                    % is_nearby:            [ size(boxes,1), numel(gt_region_idx),                   1 ]
                    % gt_conflict_score_rs: [             1, numel(gt_region_idx), numel(neg_text_ids) ]
                    box_conflict_score = bsxfun( @times, double(is_nearby), gt_conflict_score_rs );
                    conflict_score = max( box_conflict_score, [], 2 );
                    conflict_score = reshape( conflict_score, size(conflict_score,1), size(conflict_score,3) );
                    % conflict_score: [ size(boxes,1), numel(neg_text_ids) ]
                    for k = 1:size(boxes,1)
                        im.regions(k).text_id = [im.regions(k).text_id; neg_text_ids];
                        im.regions(k).label   = [im.regions(k).label; conflict_score(k,:).'];
                    end
                    % merge non-unique title (txtid-label)
                    %   make the ambiguity score as large as possible
                    %   if score is raised up to 1, it become another
                    %   ground truth label
                    im.regions = nldetMapAndUniqueTitle( im.regions, [], 'max' );
                    im         = nldetMapAndUniqueTitle( im, [], 'max' );
                otherwise
                    error( 'Invalid conflicting policy.' );
            end
                        
            % score remapping
            if ~isempty( obj.score_remapping_func )
                for k = 1:size(boxes,1)
                    im.regions(k).label = obj.score_remapping_func(im.regions(k).label);
                    invalid_title_idxb = isnan( im.regions(k).label );
                    % remove NaN (ignored) title
                    im.regions(k).text_id(invalid_title_idxb,:) = [];
                    im.regions(k).label(invalid_title_idxb,:)   = [];
                end
            end
            
            % put back the updated entry
            obj.D.img(img_id) = im;
            
        end

        
    end
    
    methods ( Access = public )
        
        % fetch next batch
        function varargout = next(obj)
            
            assert(nargout<=1,'Too many outputs');

            % init image structure
            obj.I = repmat(struct(),obj.batch_size,1);
            
            obj.image_sampler.join();
            
            % get preliminary data for single images
            for j = 1:obj.batch_size
                obj.fill_single(j);
            end
            
            obj.image_sampler.prefetch();
            
            % fuse single images to batch: accumulate and re-index text across images
            obj.fuse_batch();

            % set up outputs
            if isempty(obj.I)
                obj.D = [];
                obj.iters_in_epoch = [];
                obj.processing_level = obj.PLEVEL_INVALID;
            else            
                obj.D = struct( 'img', {obj.I}, 'text_pool', {obj.T} );
                obj.iters_in_epoch = cat(1,obj.I.iter_in_epoch);
                obj.I = []; obj.T = []; % clear tmp variables
                obj.processing_level = obj.PLEVEL_GET;
            end
            if nargout==1
                varargout = {obj.D};
            end
            
        end
        
        function done( obj )
            assert( obj.processing_level < obj.PLEVEL_DONE, 'Wrong processing level' );
            if obj.image_sampler.use_unique_guard
                for j = 1:numel(obj.iters_in_epoch)
                    obj.image_sampler.atomic_indexer.done( obj.iters_in_epoch(j) );
                end
            end
            obj.processing_level = obj.PLEVEL_DONE;
        end
        
        % add more boxes (from region proposal) to the current batch
        %   boxes should be in tlbr: [y1,x1,y2,x2]
        function add_boxes( obj, boxes )
            
            assert( obj.processing_level <= obj.PLEVEL_ADD_BOX, 'Wrong processing level' );
            if isempty(obj.D), return; end
            
            numImg = numel(obj.D.img);
            assert( numImg == obj.batch_size, 'current image cache is wrong' );
            assert( iscell(boxes) && numel(boxes) == numImg, ...
                'boxes must be a cell array have the same number of elements as the batch size' );
            
            default_region_struct = struct('box',[], 'text_id',zeros(0,1), 'label',zeros(0,1), ...
                'box_rank', inf, 'is_gt', false, 'gt_source', [], 'is_added', 0, 'nearby_gt', zeros(1,0));
            
            for j = 1:numImg
                numBoxes = size( boxes{j}, 1 );
                if numBoxes<1, continue; end
                cnRegion = numel(obj.D.img(j).regions);
                cur_max_rank = max( [ 0, obj.D.img(j).regions.box_rank ] );
                aidx = (cnRegion+1):(cnRegion+numBoxes);
                box_cell_j = mat2cell( boxes{j}, ones(numBoxes,1), 4 );
                obj.D.img(j).regions(aidx) = default_region_struct;
                [ obj.D.img(j).regions(aidx).box ] = deal(box_cell_j{:});
                [ obj.D.img(j).regions(aidx).is_added ] = rdeal( true );
                box_rank = cur_max_rank+(1:numBoxes);
                box_rank = num2cell(box_rank);
                [ obj.D.img(j).regions(aidx).box_rank ] = deal( box_rank{:} );
            end
            
            obj.processing_level = obj.PLEVEL_ADD_BOX;
            
        end
        
        % propagate text annotations
        function propagate_text( obj )
            
            assert( obj.processing_level < obj.PLEVEL_PROP_TEXT, 'Wrong processing level' );
            if isempty(obj.D), return; end
            
            % clean up neg against gt
            % propagate positive to jittered (may be use structured loss??)
            % propagete region gt as negative to other part of the image (need verify)
            
            for j = 1:obj.batch_size
                obj.propagate_text_single(j);
            end
            obj.processing_level = obj.PLEVEL_PROP_TEXT;
            
        end
        
        % feedback which box is hard, which box is easy
        function feedback( obj, pair_scores )
            
            assert( obj.processing_level < obj.PLEVEL_FEEDBACK, 'Wrong processing level' );
            
            if isempty(obj.D), return; end
            
            % update text augmenter
            feedback_idxb = [obj.taug_list.feedback];
            if any( [obj.taug_list.feedback] )
                
                % pair_scores = obj.standardize_current_pair_scores( pair_scores );
                
                numImagesInIter = numel( obj.D.img );
                for j = 1:numImagesInIter
                    curimage_entry_idxb = (codebook_cols( ...
                        pair_scores, {'image_id'} ) == j );
                    cur_pair_scores = struct( 'cols', { pair_scores.cols(2:end) }, ...
                        'dict', { pair_scores.dict(curimage_entry_idxb,2:end) } );
                    curD = struct('img',{obj.D.img(j)},'text_pool',{obj.D.text_pool});
                    augT_im = struct('order', curD.img.order, 'position', curD.img.position);
                    
                    feedback_idx = find(feedback_idxb);
                    for k1 = 1:length(feedback_idx)
                        k = feedback_idx(k1);
                        obj.taug_list(k).taug.feedback( augT_im, cur_pair_scores, curD );
                    end
                end
            end
            
            % potentially update bounding box augmenter
            %  *** implement it when needed
            %
            
            obj.processing_level = obj.PLEVEL_FEEDBACK;
            
        end
        
        function S1 = standardize_current_pair_scores( obj, S0 )
            
            if isempty(obj.D), 
                S1 = obj.standardize_pair_scores( [], S0 );
                return; 
            end
            
            S1 = obj.standardize_pair_scores( obj.D.img, S0 );
        end
        
    end
    
    
    methods (Static, Access=public)
                
        % standardize pairwise scores
        %  -- standard format (output format): 
        %        N by 4 matrix, each row [image_id, region_id, text_id, score]
        %  -- compact format: 
        %        N by 1 matrix, [image_id, region_id, text_id] are implicitly from obj.D
        %  -- structured format: 
        %        a cell array, each element for one image,
        %        for each image, a cell array, each element for one region
        %        for each region, a vector of scores which should be consistent with obj.D
        %
        function S1 = standardize_pair_scores( I, S0 )
            
            if isstruct(I) && isscalar(I) && isfield(I,'img') && isfield(I,'text_pool')
                I = I.img;
            end

            if isstruct(S0)
                if isfield( S0, 'dict' )
                    S0 = S0.dict;
                else
                    error('Unrecognized format');
                end
            end
            
            if isempty(I)
                S = zeros(0,4);
            elseif strcmp(S0,'gt')
                S = cell(numel(I),1);
                for j = 1:numel(I)
                    numRegions_j = numel(I(j).regions);
                    S{j} = cell(numRegions_j,1);
                    for k = 1:numRegions_j
                        ind = [ I(j).regions(k).text_id, ...
                            I(j).regions(k).label ];
                        ind(:,3) = j; ind(:,4) = k;
                        ind = ind(:,[3 4 1 2]);
                        S{j}{k} = ind;
                    end
                    S{j} = cat(1, zeros(0,4), S{j}{:} );
                end
                S = cat(1, S{:} );
            elseif isnumeric( S0 )
                if isvector(S0)
                    n = numel(S0);
                    S = zeros( n, 4 );
                    S(:,4) = reshape( S0, n, 1 );
                    i = 0;
                    for j = 1:numel(I)
                        for k = 1:numel(I(j).regions)
                            ind = I(j).regions(k).text_id;
                            ind(:,2) = j; ind(:,3) = k;
                            ind = ind(:,[2 3 1]);
                            S(i+1:size(ind,1),1:3) = ind;
                            i = i + size(ind,1);
                        end
                    end
                    assert( i==n, 'mismatched length' );
                else
                    assert( size(S0,2)==4 && ismatrix(S0), 'Unrecongized format' );
                    S = S0;
                end
            elseif iscell(S0)
                assert( isvector(S0) && numel(S0) == numel(I), ...
                    'mismatched image number' );
                S = cell(numel(S0),1);
                for j = 1:numel(I)
                    assert( isvector(S0{j}) && numel(S0{j}) == ...
                        numel(I(j).regions), 'mismatched region number' );
                    numRegions_j = numel(I(j).regions);
                    S{j} = cell(numRegions_j,1);
                    for k = 1:numRegions_j
                        assert( isvector(S0{j}{k}) && ...
                            numel(S0{j}{k}) == numel( I(j).regions(k).text_id ), ...
                            'mismatched title (txtid-label) number' );
                        ind = I(j).regions(k).text_id;
                        ind(:,2) = j; ind(:,3) = k;
                        ind = ind(:,[2 3 1]);
                        S{j}{k} = [ind; vec(S0{j}{k})];
                    end
                    S{j} = cat(1, zeros(0,4), S{j}{:} );
                end
                S = cat(1, S{:} );
            else
                error('Unrecongized format');
            end
            
            S1 = struct();
            S1.cols = {'image_id','region_id','text_id','score'};
            S1.dict = S;
            
        end
        
    end
    
    methods (Access = public)
        
        function snapshot( obj, snapshot_dir )
            
            mkdir_p(snapshot_dir); 
            
            % state of image sampler
            obj.module_io( obj.image_sampler, 'snapshot', ...
                fullfile( snapshot_dir, 'image_sampler' ) );
            
            % text augmenter states
            taug_folder = fullfile( snapshot_dir, 'text_augmenters' );
            mkdir_p(taug_folder);
            for k = 1:numel( obj.taug_list )
                ta = obj.taug_list(k);
                obj.module_io( ta.taug, 'snapshot', ...
                    fullfile( snapshot_dir, 'text_augmenters', ta.id_str ) );
            end
            
        end

        function restore( obj, snapshot_dir )
            
            obj.try_restore( snapshot_dir, 1 );
            
        end
        
        function try_restore( obj, snapshot_dir, no_try )
            
            mkdir_p(snapshot_dir); 
            
            % state of image sampler
            obj.module_io( obj.image_sampler, 'try_restore', ...
                fullfile( snapshot_dir, 'image_sampler' ) );
            
            % text augmenter states
            taug_folder = fullfile( snapshot_dir, 'text_augmenters' );
            mkdir_p(taug_folder);
            for k = 1:numel( obj.taug_list )
                ta = obj.taug_list(k);
                obj.module_io( ta.taug, 'try_restore', ...
                    fullfile( snapshot_dir, 'text_augmenters', ta.id_str ) );
            end
            
        end
        
    end
    
    methods ( Static, Access = protected )
        
        function module_io( module_obj, io_func_str, module_output_dir )
            if ismethod(module_obj, io_func_str)
                io_func = eval( sprintf( '@(varargin) module_obj.%s(varargin{:})', ...
                    io_func_str ) );
                io_func( module_output_dir );
            end
        end
        
    end
    
    methods (Static, Access = public)
        
        function visualize_annotations( output_folder, D, ids_in_minibatch )
            if nargin<3 || isempty( ids_in_minibatch )
                ids_in_minibatch = 1:numel(D.img);
            end
            if numel( ids_in_minibatch ) > 1
                for k = 1:numel(ids_in_minibatch)
                    nldetBatchSampler.visualize_annotations( ...
                        sprintf( '%s-%d', output_folder, k ), ...
                        D, ids_in_minibatch(k) );
                end
                return;
            end
            
            mkdir_p( output_folder );
            
            I = D.img(ids_in_minibatch);
            
            if isfield(D,'pair')
                L = D.pair;
                L.dict( ~( codebook_cols( L, 'image_id' ) ...
                    == ids_in_minibatch ), : ) = [];
            else
                D1= D; D1.img = I;
                L = nldetBatchSampler.standardize_pair_scores( D1, 'gt' );
            end
            
            [utext_ids, ~, text_uids] = unique( codebook_cols( L, 'text_id' ) );
            
            C = codebook_cols( L, {'region_id','score'} );
            tfig = figure('Visible','off','Color','none');
            tfig_cleanup = onCleanup( @() delete(tfig) );
            himg = imshow(I.im);
            himg.UserData = struct();
            if isfield(I,'meta') && isfield(I.meta,'url')
                himg.UserData.url = I.meta.url;
            elseif isfield(I,'im_path')
                himg.UserData.url = I.im_path;
            end
            taxes = gca;
            for k = 1:numel(utext_ids)
                cur_text_id = utext_ids(k);
                chosen_idxb = ( text_uids == k );
                chosen_idx  = find(chosen_idxb);
                
                B = zeros(numel(chosen_idx),5);
                for j = 1:length(chosen_idx)
                    i = chosen_idx(j);
                    B(j,1:4) = I.regions(C(i,1)).box;
                end
                B(:,5) = C(chosen_idxb,2);
                
                G = findobj( taxes );
                nonremovalbe_idxb = arrayfun( @(a) ismember( a.Type, {'image','axes'} ) , G );
                delete( G(~nonremovalbe_idxb) );
                set(0,'CurrentFigure',  tfig);
                set(tfig,'CurrentAxes', taxes);
                % show negative boxes
                show_bboxes( [], B(B(:,5)<=0,1:4), [], {'red','black','black'} );
                % show ambigous boxes
                show_bboxes( [], B(B(:,5)>0 & B(:,5)<1,1:4), [], {'yellow','white','black'} );
                % gt boxes
                show_bboxes( [], B(B(:,5)>=1,1:4), [], {'green','white','black'} );
                
                % dump to html
                t = D.text_pool(cur_text_id);
                t_str = sprintf('%d : %s', t.source, t.phrase );
                
                fig2html_param = struct();
                fig2html_param.html_head     = sprintf('<title>%s</title>', t_str);
                fig2html_param.header_html   = sprintf('<p>%s</p>', t_str );
                fig2html_param.figure_prefix = 'figures/$$-';
                switch_page_html = '';
                if k>1
                    switch_page_html = [switch_page_html, sprintf(' [<a href="%d.html">previous</a>] ', k-1)];
                end
                if k<numel(utext_ids)
                    switch_page_html = [switch_page_html, sprintf(' [<a href="%d.html">next</a>] ', k+1)];
                end
                fig2html_param.footer_html = sprintf('<div style="text-align: center">%s</div>\n', switch_page_html);
                
                fig2html( tfig, fullfile(output_folder, sprintf('%d.html',k) ), fig2html_param );
                
            end
            
        end
        
    end
    
    methods (Access=public)
        
        function visualize_current_annotations( obj, output_folder, ids_in_minibatch )
            if nargin<3, ids_in_minibatch = []; end
            obj.visualize_annotations( output_folder, obj.D, ids_in_minibatch );
        end
        
    end
    
end
