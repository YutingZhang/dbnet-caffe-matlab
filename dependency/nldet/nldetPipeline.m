classdef nldetPipeline < handle
    
    properties (GetAccess = public, SetAccess = protected)
        
        rand_stream
        
        batch_sampler
        batch_stacker   % stack images to batch blob
        conv_feat_net
        region_proposal_net
        region_feat_net
        text_feat_net
        pair_net

        restore_solver_batch_sampler       = 1
        restore_solver_conv_feat_net       = 1
        restore_solver_region_proposal_net = 1
        restore_solver_region_feat_net     = 1
        restore_solver_text_feat_net       = 1
        restore_solver_pair_net            = 1
        
        need_update_conv_feat_net
        need_update_region_proposal_net
        need_update_region_feat_net
        need_update_text_feat_net
        need_update_pair_net
        
        % imboxes    % [img_id, y1, x1, y2, x2]
        % imbox_inds % [img_id, region_id]
        % phrases    % all phrases
        pair_codebook % standardized score codebook (both tested and gt)
        
        formatted_test_results
        
        is_train        % train or test
        train_state = struct();
        
        last_iter_time_cost;
        
        param
        
        raw_iter
        
        need_backward_conv_feat_net
        need_backward_region_proposal_net
        need_backward_region_feat_net
        need_backward_text_feat_net
        need_backward_pair_net
        
    end
    
    properties (GetAccess = public, SetAccess = protected)
        
        batch_data
        conv_feat
        proposed_boxes
        region_feat
        text_feat
        pair_output
        
    end
    
    methods (Access = public)
        
        function obj = nldetPipeline( mode, batch_sampler, ...
                batch_stacker, conv_feat_net, region_proposal_net, ...
                region_feat_net, text_feat_net, pair_net, PARAM )
            
            if ~exist('PARAM','var') || isempty(PARAM)
                PARAM = struct();
            end
            Pdef = struct();
            Pdef.rand_seed = 49582;
            Pdef.visualization_path = []; % if specified visualize the boxes
            Pdef.region_proposal_input = []; % must specified
            Pdef.region_proposal_top_num    = inf; % keep how many top proposals
            Pdef.region_proposal_random_num = 0;   % how many more random proposals
            Pdef.remove_orphan_phrases_in_results = false; % remove orphan gt
            PARAM = xmerge_struct(Pdef,PARAM);
            
            assert( ismember(PARAM.region_proposal_input,{'none','id','image','feature'}), ...
                'Invalid specification of PARAM.region_proposal_input' );
            
            obj.param = PARAM;
            
            obj.rand_stream = RandStream('mrg32k3a','seed',obj.param.rand_seed);
            
            switch lower(mode)
                case 'train'
                    obj.is_train = 1;
                case 'test'
                    obj.is_train = 0;
                otherwise
                    error( 'mode should be TRAIN or TEST' );
            end
            
            obj.need_update_conv_feat_net       = obj.is_train;
            obj.need_update_region_proposal_net = obj.is_train;
            obj.need_update_region_feat_net     = obj.is_train;
            obj.need_update_text_feat_net       = obj.is_train;
            obj.need_update_pair_net            = obj.is_train;
            
            obj.batch_sampler = batch_sampler;
            obj.batch_stacker = batch_stacker;
            obj.conv_feat_net = conv_feat_net;
            obj.region_proposal_net = region_proposal_net;
            obj.region_feat_net = region_feat_net;
            obj.text_feat_net = text_feat_net;
            obj.pair_net      = pair_net;
            
            obj.batch_data = struct( 'im_blob', [], ...
                'trans_mat', [] );
            empty_blob = struct('data',[],'diff',[]);
            obj.conv_feat      = empty_blob;
            obj.proposed_boxes = empty_blob;
            obj.region_feat    = empty_blob;
            obj.text_feat      = empty_blob;
            obj.pair_output    = empty_blob;
            
            obj.train_state = struct();
            obj.train_state.self_iter = 0;
            obj.train_state.base_iter = 0;
            
            obj.raw_iter = 0;
            
            obj.update_backward_flags();
                        
            % set memory econ
            if ismethod( obj.pair_net, 'SetMemoryEcon' )
                obj.pair_net.SetMemoryEcon( true );
            end
            if ismethod( obj.region_feat_net, 'SetMemoryEcon' )
                obj.region_feat_net.SetMemoryEcon( true );
            end
            if ismethod( obj.region_proposal_net, 'SetMemoryEcon' )
                obj.region_proposal_net.SetMemoryEcon( true );
            end
            if ismethod( obj.conv_feat_net, 'SetMemoryEcon' )
                obj.conv_feat_net.SetMemoryEcon( true );
            end
            if ismethod( obj.text_feat_net, 'SetMemoryEcon' )
                obj.text_feat_net.SetMemoryEcon( true );
            end
            
        end
        
        function update_backward_flags( obj )
            % figure out if backward is needed for each block
            
            obj.need_backward_conv_feat_net = ...
                obj.need_update_conv_feat_net; 
            
            obj.need_backward_region_proposal_net = ... 
                ( strcmp(obj.param.region_proposal_input,'feature') && ...
                obj.need_update_region_proposal_net ) || ...
                obj.need_backward_conv_feat_net;
            % remark: if region_proposal_net depends on conv_feat_net
            
            obj.need_backward_region_feat_net = ...
                obj.need_update_region_feat_net || ...
                obj.need_backward_region_proposal_net || ...
                obj.need_backward_conv_feat_net;
            
            obj.need_backward_text_feat_net = ...
                obj.need_update_text_feat_net;
            
            obj.need_backward_pair_net = ...
                obj.need_update_pair_net || ...
                obj.need_backward_region_feat_net || ...
                obj.need_backward_text_feat_net;

            % figure out if the block input diff need to be outputed
            if ismethod( obj.pair_net, 'ToggleInputDiff' )
                % im pathway
                obj.pair_net.ToggleInputDiff( 1, ...
                    obj.need_backward_region_feat_net);
                obj.pair_net.ToggleInputDiff( 2, ...
                    obj.need_backward_text_feat_net);
                if obj.is_train
                    obj.pair_net.ToggleInputDiff( 3, false );
                end
            end
            if ismethod( obj.region_feat_net, 'ToggleInputDiff' )
                % im pathway
                obj.region_feat_net.ToggleInputDiff( 1, ...
                    obj.need_backward_conv_feat_net);
                obj.region_feat_net.ToggleInputDiff( 2, ...
                    obj.need_backward_region_proposal_net);
            end
            if ismethod( obj.region_proposal_net, 'ToggleInputDiff' )
                % **** complicated, not implemented
            end
            if ismethod( obj.conv_feat_net, 'ToggleInputDiff' )
                obj.conv_feat_net.ToggleInputDiff( 1, false );
            end
            if ismethod( obj.text_feat_net, 'ToggleInputDiff' )
                obj.text_feat_net.ToggleInputDiff( 1, false );
            end

        end
        
        function toggle_block_updatable( obj, block_name, val )
            flag_name = sprintf( 'need_update_%s', block_name );
            assert( isprop( obj, flag_name ), 'no such property to set' );
            ss = substruct( '.', flag_name );
            if nargin<3
                val = ~subsref( obj, ss );
            end
            [~] = subsasgn( obj, ss, val ); % obj will be modified
            obj.update_backward_flags();
        end

        function toggle_block_restore_solver( obj, block_name, val )
            flag_name = sprintf( 'restore_solver_%s', block_name );
            assert( isprop( obj, flag_name ), 'no such property to set' );
            ss = substruct( '.', flag_name );
            if nargin<3
                val = ~subsref( obj, ss );
            end
            [~] = subsasgn( obj, ss, val ); % obj will be modified
        end

        
        function step( obj )
            
            step_timer = tic;
            
            if obj.is_train
                obj.raw_iter = floor(obj.train_state.self_iter)+1;
                
                % save gpu memory
                obj.clear_cache();
            else
                obj.raw_iter = obj.raw_iter + 1;
            end
            
            % sample data and annotations
            D = obj.batch_sampler.next();
            
            if isempty(D),
                tic_toc_print('Data ran out.\n');
                return;
            end
            if obj.is_train
                while isempty(D.img(1).regions)
                    D = obj.batch_sampler.next();
                end
            end
            if isempty(D),
                tic_toc_print('Data ran out.\n');
                return;
            end
            
            if ~isempty( obj.param.visualization_path )
                obj.batch_sampler.visualize_current_annotations( ...
                    fullfile( obj.param.visualization_path, 'viz_sample_gt', sprintf('iter_%d',obj.raw_iter) ) );
            end
            % D0 = D; % for debug
            
            % resize and stack images, and get image2feature transformation
            if obj.raw_iter==1, t_raw = tic_print( 'first_iter: batch_sampler : ' ); end
            I = {D.img.im};
            obj.batch_stacker.SetInputData(I);
            obj.batch_stacker.Forward();
            [obj.batch_data.im_blob, obj.batch_data.trans_mat] = ...
                obj.batch_stacker.GetOutputData();
            if ismethod(obj.conv_feat_net,'ImageToFeatureTransform') || ...
                    isprop(obj.conv_feat_net,'ImageToFeatureTransform')
                ib2ftT = obj.conv_feat_net.ImageToFeatureTransform;
            else
                ib2ftT = [eye(2),zeros(2,1)];
            end
            % remark ib2ftT is in [x,y,1]
            im2ftT = cell(1,length(D.img));
            for k = 1:length(D.img)
                im2ftT{k} = compact_trans( homo_trans( ib2ftT ) * ...
                    homo_trans( obj.batch_data.trans_mat(:,:,k) ) );
            end
            if obj.raw_iter==1, toc_print( t_raw ); end
            
            % extract convolutional features
            if obj.raw_iter==1, t_raw = tic_print( 'first_iter: conv_feat_net : ' ); end
            obj.conv_feat_net.SetInputData(obj.batch_data.im_blob);
            obj.conv_feat_net.Forward();
            obj.conv_feat.data = obj.conv_feat_net.GetOutputData();
            if obj.raw_iter==1, toc_print( t_raw ); end

            % region proposal
            if obj.raw_iter==1, t_raw = tic_print( 'first_iter: region_proposal_net : ' ); end
            rpn_input = {};
            %if ~isfield( obj.param, 'region_proposal_input' ) % **** temporary fallback support  (remove it later)
            %    obj.param.region_proposal_input = 'id';
            %end
            switch obj.param.region_proposal_input
                case 'none'
                case 'id'
                    img_ids = -ones(1, numel(D.img), 'single');
                    if isfield(D.img,'meta') % get image id if possible
                        img_meta = [D.img.meta];
                        if isfield(img_meta, 'image_id')
                            if isnumeric(img_meta(1).image_id)
                                img_ids = single( [ img_meta.image_id ] );
                            else
                                img_ids = { img_meta.image_id };
                            end
                        end
                    end
                    rpn_input = {img_ids};
                case 'image'
                    rpn_input = {I};
                case 'feature'
                    im_sizes = zeros(numel(I),2);
                    scaled_im_sizes = zeros(numel(I),2);
                    for k = 1:numel(I)
                        im_sizes(k,:)        = [size(I{k},1),size(I{k},1)];
                        scaled_im_sizes(k,:) = [im_sizes(k,1)*obj.batch_data.trans_mat(2,2), ...
                            im_sizes(k,2)*obj.batch_data.trans_mat(1,1)];
                    end
                    rpn_input = {obj.conv_feat.data, im_sizes, scaled_im_sizes};
                otherwise
                    error( 'internal error: unrecognized region_proposal_input' );
            end
            obj.region_proposal_net.SetInputData( rpn_input{:} );
            % remark: can start from image, or use cached boxes when needed
            %   or get results from the conv_feat_net
            obj.region_proposal_net.Forward();
            obj.proposed_boxes.data = obj.region_proposal_net.GetOutputData();
            % remark: the output data is 1-based, [image_id,y1,x1,y2,x2]
            if obj.raw_iter==1, toc_print( t_raw ); end
            
            % generate full annotations
            if obj.raw_iter==1, t_raw = tic_print( 'first_iter: generate annotations : ' ); end
            propB = cell(1,numel(D.img));
            propBchosen  = cell(1,numel(D.img));
            pbChosenIdxb = cell(1,numel(D.img));
            for k = 1:numel(D.img)
                
                per_image_idxb = (obj.proposed_boxes.data(:,1) == k);
                propB{k} = obj.proposed_boxes.data(per_image_idxb,[2 3 4 5]);
                
                numBoxes_k = size(propB{k},1);
                pbChosenIdxb_k = false( numBoxes_k,1 );
                
                % sampling proposed boxes
                numTopBoxes_k = min(numBoxes_k,obj.param.region_proposal_top_num);
                pbChosenIdxb_k(1:numTopBoxes_k) = true;
                numRemainBoxes_k = numBoxes_k-numTopBoxes_k;
                idxRandomBoxes_k = numTopBoxes_k + ...
                    obj.rand_stream.randperm( ...
                    numRemainBoxes_k, ...
                    min(numRemainBoxes_k,obj.param.region_proposal_random_num) );
                pbChosenIdxb_k(idxRandomBoxes_k) = true;

                pbChosenIdxb{k} = pbChosenIdxb_k;
                propBchosen{k} = propB{k}(pbChosenIdxb_k,:);
            end
            
            obj.batch_sampler.add_boxes( propBchosen );
            obj.batch_sampler.propagate_text();
            D = obj.batch_sampler.current();
            L = obj.batch_sampler.standardize_current_pair_scores( 'gt' );
            
            if ~isempty( obj.param.visualization_path )
                obj.batch_sampler.visualize_current_annotations( ...
                    fullfile( obj.param.visualization_path, 'viz_sample_aug', sprintf('iter_%d',obj.raw_iter) ) );
            end
            
            if size(L.dict,1)==0
                obj.make_pair_codebook(L,[]);
                if ~obj.is_train
                    obj.format_test_results();
                end
                obj.batch_sampler.done();
                obj.last_iter_time_cost = toc(step_timer);
                fprintf( '[No box present]' );
                obj.clear_cache();
                return;
            end
            
            if obj.is_train
                % may be useful for batch normalization
                L.dict = L.dict(randperm(size(L.dict,1)),:);
            end
            
            if ~isempty( obj.param.visualization_path )
                % viz finally selected samples
                Dviz = D;
                Dviz.pair = L;
                obj.batch_sampler.visualize_annotations( ...
                    fullfile( obj.param.visualization_path, 'viz_sample_finalized', ...
                    sprintf('iter_%d',obj.raw_iter) ) , Dviz );
            end
            
            box_ids = codebook_cols( L, {'image_id','region_id'} );
            [ubox_ids, ~, box_uids] = unique(box_ids, 'rows', 'stable');
            text_uids = codebook_cols( L, {'text_id'} );
            
            imB = zeros(size(ubox_ids,1),5,'single');
            for k = 1:size(imB,1)
                imB(k,:) = [ubox_ids(k,1), ...
                    D.img( ubox_ids(k,1) ).regions( ubox_ids(k,2) ).box];
            end
            ftB = zeros( size(imB), 'single');
            for k = 1:length(D.img) % transform boxes for ROI pooling
                bidx_in_im = ( imB(:,1) == k );
                X = imB( bidx_in_im, [3 2 5 4] ).'; % y,x --> x,y
                X = reshape( X, 2, size(X,2)*2 );
                X = X-1;
                X(3,:) = 1;
                Y = homo_trans( im2ftT{k} ) * X;
                Y = Y(1:2,:)./Y([3 3],:);
                Y = Y + 1;
                Y = reshape( Y, 4, size(Y,2)/2 );
                ftB( bidx_in_im, [3 2 5 4] ) = Y.'; % x,y --> y,x
                %if any(isinf(Y(:))) || any(isnan(Y(:)))
                %    keyboard;
                %end
            end
            ftB(:,1) = imB(:,1);
            clear X Y
            
            if obj.raw_iter==1, toc_print( t_raw ); end
            
            % extract regional features
            if obj.raw_iter==1, t_raw = tic_print( 'first_iter: region_feat_net : ' ); end
            obj.region_feat_net.SetInputData( ...
                obj.conv_feat.data, ftB );
            obj.region_feat_net.Forward();
            obj.region_feat.data = obj.region_feat_net.GetOutputData();
            if obj.raw_iter==1, toc_print( t_raw ); end
            
            obj.conv_feat.data = []; % save GPU memory
            
            % forward text network to extract text features
            if obj.raw_iter==1, t_raw = tic_print( 'first_iter: text_feat_net : ' ); end
            obj.text_feat_net.SetInputData( {D.text_pool.phrase} );
            obj.text_feat_net.Forward();
            obj.text_feat.data = obj.text_feat_net.GetOutputData();
            if obj.raw_iter==1, toc_print( t_raw ); end
            
            % forward/backward text and image pair
            if obj.raw_iter==1, t_raw = tic_print( 'first_iter: pair_net : ' ); end
            gt_sim = codebook_cols(L, {'score'});
            gt_sim = vec(gt_sim).';
            region_feat_blob = {obj.region_feat.data, box_uids};
            text_feat_blob   = {obj.text_feat.data, text_uids};
            if obj.is_train
                obj.pair_net.SetInputData( ...
                    region_feat_blob, text_feat_blob, gt_sim );
                % for efficiency, do forward and backward together
                % No need to set diff. Loss functions should be in pair_net
                % obj.pair_net.SetOutputDiff();
                if obj.need_backward_pair_net
                    obj.pair_net.ForwardBackward();
                    [obj.region_feat.diff, obj.text_feat.diff] = obj.pair_net.GetInputDiff();
                else
                    obj.pair_net.Forward();
                end
            else
                obj.pair_net.SetInputData( ...
                    region_feat_blob, text_feat_blob );
                obj.pair_net.Forward();
            end
            obj.pair_output.data = obj.pair_net.GetOutputData();
            if obj.raw_iter==1, toc_print( t_raw ); end

            obj.text_feat.data   = []; % save GPU memory
            obj.region_feat.data = []; % save GPU memory
            
            if obj.is_train
                [loss_pos, loss_neg, pred_scores] = obj.output();
                loss_pos = gather(loss_pos);
                loss_neg = gather(loss_neg);
            else
                pred_scores = obj.output();
            end
            pred_scores = gather(pred_scores);
            
            obj.make_pair_codebook(L,pred_scores);
            
            if ~obj.is_train
                obj.last_iter_time_cost = toc(step_timer);
                obj.format_test_results();
                
                obj.batch_sampler.done();
                return;
            end
            
            pre_iter = floor(obj.train_state.self_iter);
            fprintf( 'iter: %d\tloss(pos): %g\tloss(neg): %g\t', ...
                pre_iter+1, loss_pos, loss_neg(1) );
            if numel(loss_neg)>1
                assert(numel(loss_neg)==2, 'unrecognized loss output');
                fprintf( 'loss(loc): %g\t', loss_neg(2) );
            end
            obj.train_state.self_iter = pre_iter + 0.05;
            
            % update pair_net
            if obj.need_update_pair_net
                obj.pair_net.Update();
            end
            
            % backward text features
            if obj.need_backward_text_feat_net
                obj.text_feat_net.SetOutputDiff( obj.text_feat.diff );
                obj.text_feat_net.Backward();
                obj.text_feat.diff = []; % save GPU memory
            end
            
            % update text_feat_net
            if obj.need_update_text_feat_net
                obj.text_feat_net.Update();
            end
            
            % backward regional features
            if obj.need_backward_region_feat_net
                obj.region_feat_net.SetOutputDiff( obj.region_feat.diff );
                obj.region_feat_net.Backward();
                % remark: if dependent network, Backward computation may be
                %           non-avoidable, but it can be logically disabled
                %           by not setting the OutputDiff
                [obj.conv_feat.diff, obj.proposed_boxes.diff] = obj.region_feat_net.GetInputDiff();
                obj.region_feat.diff = []; % save GPU memory
            end
            
            % update region_feat_net
            if obj.need_update_region_feat_net
                obj.region_feat_net.Update();
            end
            
            % backward region proposal
            if obj.need_backward_region_proposal_net
                obj.region_proposal_net.SetOutputDiff( obj.proposed_boxes.diff );
                obj.region_proposal_net.Backward();
                if strcmp(obj.param.region_proposal_input,'feature')
                    conv_feat_diff_2 = obj.region_proposal_net.GetInputDiff();
                    if obj.need_backward_region_feat_net
                        obj.conv_feat.diff = obj.conv_feat.diff + conv_feat_diff_2;
                    else
                        obj.conv_feat.diff = conv_feat_diff_2;
                    end
                end
                obj.proposed_boxes.diff = [];  % save GPU memory
            end
            
            % update region_proposal_net
            if obj.need_update_region_proposal_net
                obj.region_proposal_net.Update();
            end
            
            % backward convolutional features
            if obj.need_backward_conv_feat_net
                obj.conv_feat_net.SetOutputDiff( obj.conv_feat.diff );
                obj.conv_feat_net.Backward();
                obj.conv_feat.diff = []; % save GPU memory
            end
            
            % update conv_feat_net
            if obj.need_update_conv_feat_net, 
                obj.conv_feat_net.Update();
            end

            % update training iter
            obj.train_state.self_iter = pre_iter + 1;
            
            % feedback batch-sampler
            obj.batch_sampler.feedback( obj.pair_codebook );
            
            % done for this iter
            obj.batch_sampler.done();
            
            obj.last_iter_time_cost = toc(step_timer);
            toc(step_timer);
            
        end
        
        function varargout = output( obj )
            varargout = cell(1,max(1,nargout));
            [varargout{:}] = obj.pair_net.GetOutputData();
        end
        
        function iter = train_self_iter( obj )
            iter = obj.train_state.self_iter;
        end

        function iter = train_total_iter( obj )
            iter = obj.train_state.self_iter + obj.train_state.base_iter;
        end
        
        
        function save( obj, output_dir )
            
            fprintf('\n');
            mkdir_p( output_dir );
            obj.block_io( 'conv_feat_net',       'Save', output_dir );
            obj.block_io( 'region_proposal_net', 'Save', output_dir );
            obj.block_io( 'region_feat_net',     'Save', output_dir );
            obj.block_io( 'text_feat_net',       'Save', output_dir );
            obj.block_io( 'pair_net',            'Save', output_dir );
            
            if obj.is_train
                save_from_struct( fullfile(output_dir, 'train_state.mat'), ...
                    obj.train_state );
                obj.block_io( 'batch_sampler', 'snapshot', output_dir );
            end
        end
        
        function load( obj, input_dir )
            fprintf('\n');
            if obj.restore_solver_conv_feat_net
                obj.block_io( 'conv_feat_net', 'TryLoad_Solver', input_dir );
            else
                obj.block_io( 'conv_feat_net', 'TryLoad', input_dir );
            end
            if obj.restore_solver_region_proposal_net
                obj.block_io( 'region_proposal_net', 'TryLoad_Solver', input_dir );
            else
                obj.block_io( 'region_proposal_net', 'TryLoad', input_dir );
            end
            if obj.restore_solver_region_feat_net
                obj.block_io( 'region_feat_net',     'TryLoad_Solver', input_dir );
            else
                obj.block_io( 'region_feat_net',     'TryLoad', input_dir );
            end
            if obj.restore_solver_text_feat_net
                obj.block_io( 'text_feat_net',       'TryLoad_Solver', input_dir );
            else
                obj.block_io( 'text_feat_net',       'TryLoad', input_dir );
            end
            if obj.restore_solver_pair_net
                obj.block_io( 'pair_net',            'TryLoad_Solver', input_dir );
            else
                obj.block_io( 'pair_net',            'TryLoad', input_dir );
            end
            
            if obj.is_train
                train_state_fn = fullfile(input_dir, 'train_state.mat');
                if exist( train_state_fn, 'file' )
                    train_state1 = load( fullfile(input_dir, 'train_state.mat') );
                    obj.train_state = xmerge_struct( obj.train_state, train_state1 );
                end
                obj.train_state.self_iter = ceil( obj.train_state.self_iter );
                if obj.restore_solver_batch_sampler
                    obj.block_io( 'batch_sampler', 'try_restore', input_dir );
                end
            end

        end
        
        function load_pretrained( obj, input_dir )
            obj.load( input_dir );

            % transfer self_iter to base_iter
            obj.train_state.base_iter = obj.train_state.base_iter+obj.train_state.self_iter;
            obj.train_state.self_iter = 0;
        end
        
    end
    
    methods ( Access = protected )
        
        function make_pair_codebook( obj, L, pred_scores )
            RESULT = gather(L);
            RESULT.cols{end}   = 'label';
            RESULT.cols{end+1} = 'score';
            RESULT.dict = gather(RESULT.dict);
            RESULT.dict = [RESULT.dict, vec(pred_scores)];
            obj.pair_codebook = RESULT;
        end
        
        function block_io( obj, module_name, io_func_str, output_dir )
            M = subsref( obj, substruct('.', module_name) );
            if ismethod(M, io_func_str)
                fprintf( ' - %s\n', module_name );
                io_func = eval( sprintf( '@(varargin) obj.%s.%s(varargin{:})', ...
                    module_name, io_func_str ) );
                io_func( fullfile( output_dir, module_name ) );
            end
        end
        
        function format_test_results( obj )
            D = obj.batch_sampler.current();
            batch_size = numel(D.img);
            L = obj.pair_codebook;
            L = codebook_cols( L, {'image_id','region_id','text_id','label','score'} );
            R = repmat( struct( 'im_path', [], 'boxes', [], 'box_ranks', [], 'text', [], ...
                'scores', [], 'labels', [], 'is_gt', [] ), batch_size, 1 );
            for k = 1:batch_size
                
                R(k).im_path = D.img(k).im_path;
                if isfield(D.img(k),'meta')
                    R(k).meta = D.img(k).meta;
                end
                R(k).boxes = cat(1,D.img(k).regions.box);
                R(k).box_ranks = cat(1,D.img(k).regions.box_rank);
                
                Q = L( L(:,1) == k, : );
                [this_utext_ind, ~, this_text_sub] = unique( Q(:,3) );
                R(k).text = D.text_pool(this_utext_ind);
                ann_sources = [R(k).text.source];
                
                numRegions = size(R(k).boxes,1);
                numText    = numel(R(k).text);
                
                is_gt   = false(numRegions,numText); % reminder: in evaluation, any gt box should be removed
                gt_box_idx = find( [ D.img(k).regions.is_gt ] );
                for j0 = 1:length(gt_box_idx)
                    j = gt_box_idx(j0);
                    [~,text_id_j] = ismember( D.img(k).regions(j).gt_source, ann_sources );
                    is_gt(j,text_id_j) = true;
                end
                
                scores  = nan(numRegions,numText);
                labels  = nan(numRegions,numText);
                res_ind = sub2ind( [numRegions,numText], Q(:,2), this_text_sub );
                scores(res_ind) = Q(:,5);
                labels(res_ind) = Q(:,4);
                R(k).scores = scores;
                R(k).labels = labels;
                R(k).is_gt  = is_gt;
                
            end
            
            if obj.param.remove_orphan_phrases_in_results
                for k = 1:batch_size
                    is_im_gt = any( R(k).is_gt, 2 );
                    orphan_phrases_idxb = all( ...
                        isnan( R(k).scores(~is_im_gt,:) ) , 1 );
                    R(k).scores(:,orphan_phrases_idxb) = [];
                    R(k).is_gt(:,orphan_phrases_idxb)  = [];
                    R(k).labels(:,orphan_phrases_idxb) = [];
                    R(k).text(orphan_phrases_idxb) = [];
                end
            end
                        
            obj.formatted_test_results = R;
        end
        
    end
    
    methods (Static, Access=public)
        
        function visualize_results( output_folder, R, gt_boxes_for_test )
            
            if nargin<3
                gt_boxes_for_test = 0;
            end
            
            if numel(R) > 1
                for k = 1:numel(R)
                    nldetPipeline.visualize_results( ...
                        sprintf( '%s-%d', output_folder, k ), R(k), ...
                        gt_boxes_for_test );
                end
                return;
            end
            
            im = cached_file_func( @imread, R.im_path, 'images', 3000 );

            tfig = figure('Visible','off','Color','none');
            tfig_cleanup = onCleanup( @() delete(tfig) );
            himg = imshow(im);
            himg.UserData = struct();
            if isfield(R,'meta') && isfield(R.meta,'url')
                himg.UserData.url = R.meta.url;
            elseif isfield(R,'im_path')
                himg.UserData.url = R.im_path;
            end
            taxes = gca;
            
            num_boxes   = size(R.scores,1);
            num_phrases = size(R.scores,2);
            
            if ~gt_boxes_for_test
                is_gt_box_idxb = any(R.is_gt,2);
            end
            
            for k = 1:num_phrases
                
                valid_box_idxb = ~isnan( R.scores(:,k) );
                if ~gt_boxes_for_test
                    % all gt boxes (non-proposed boxes should be removed)
                    valid_box_idxb = valid_box_idxb & ~is_gt_box_idxb;
                end
                valid_box_idx  = find(valid_box_idxb);
                
                top_idx = nms( [R.boxes(valid_box_idxb,:) R.scores(valid_box_idxb,k)], 0.3 );
                top_idx = valid_box_idx(top_idx);
                top_idx(6:end) = []; % only show top-5
                gt_idx = find(R.is_gt(:,k));
                
                G = findobj( taxes );
                nonremovalbe_idxb = arrayfun( @(a) ismember( a.Type, {'image','axes'} ) , G );
                delete( G(~nonremovalbe_idxb) );
                set(0,'CurrentFigure',  tfig);
                set(tfig,'CurrentAxes', taxes);
                % show detected boxes
                for j = length(top_idx):-1:1
                    i = top_idx(j);
                    show_bboxes( [], R.boxes(i,:), R.scores(i,k), {'red','white','black'}, [], [], ...
                        sprintf('%d:', j) );
                end
                % gt boxes
                for j = length(gt_idx):-1:1
                    i = gt_idx(j);
                    show_bboxes( [], R.boxes(i,:), [], {'green','white','black'} );
                end
                
                % dump to html
                t = R.text(k);
                t_str = sprintf('%d : %s', t.source, t.phrase );
                
                fig2html_param = struct();
                fig2html_param.html_head     = sprintf('<title>%s</title>', t_str);
                fig2html_param.header_html   = sprintf('<p>%s</p>', t_str );
                fig2html_param.figure_prefix = 'figures/$$-';
                switch_page_html = '';
                if k>1
                    switch_page_html = [switch_page_html, sprintf(' [<a href="%d.html">previous</a>] ', k-1)];
                end
                if k<num_phrases
                    switch_page_html = [switch_page_html, sprintf(' [<a href="%d.html">next</a>] ', k+1)];
                end
                fig2html_param.footer_html = sprintf('<div style="text-align: center">%s</div>\n', switch_page_html);
                
                fig2html( tfig, fullfile(output_folder, sprintf('%d.html',k) ), fig2html_param );
                
            end
 
            
        end
        
        function clear_cache()
            obj.conv_feat.data   = [];
            obj.region_feat.data = [];
            obj.proposed_boxes.data = [];
            obj.text_feat.data   = [];

            obj.conv_feat.diff   = [];
            obj.region_feat.diff = [];
            obj.proposed_boxes.diff = [];
            obj.text_feat.diff   = [];
        end
        
    end
    
    methods (Access=public)

        function visualize_current_results( obj, output_folder, gt_boxes_for_test )
            if nargin>=3
                obj.visualize_results( output_folder, obj.formatted_test_results, gt_boxes_for_test );
            else
                obj.visualize_results( output_folder, obj.formatted_test_results );
            end
        end
        
    end
    
end
