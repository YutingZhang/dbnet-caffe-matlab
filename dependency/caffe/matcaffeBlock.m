classdef matcaffeBlock < handle

    properties (Access = public)
        
        aux = struct();
        
    end
    
    properties (GetAccess = public, SetAccess = private)
        
        worker
        net
        solver
        
        blob_loss_weights
        
        func_forward  = @(varargin) error('Undefined forward');
        func_backward = @(varargin) error('Undefined backward');
        func_update   = @(varargin) error('Undefined update');
        
        is_solver
        base_iter
        
        input_names
        output_names
        loss_output_idxb
        actual_output_idxb
        
        input_data
        input_index
        output_diff
        input_info
        
        backward_input_diff
        input_diff
        output_data
        
        input_data_subbatch_id
        output_diff_subbatch_id
        
        input_max_batch_sizes
        output_max_batch_sizes
        input_batch_dims
        output_batch_dims
        
        num_subbatch
        last_subbatch_ratio
        subbatch_ratios
        
        rand_stream
        
        param
        
        use_device  = 0
        
        memory_econ = 0
        
    end
    
    properties (Access = private)
        
        input_subbatch_idx
        output_subbatch_idx
        
    end
    
    methods (Access = public)
        
        function obj = matcaffeBlock( solver_or_net, PARAM )
            
            if ~exist('PARAM','var') || isempty(PARAM)
                PARAM = struct();
            end
            
            Pdef = struct();
            Pdef.batch_size_mult = 1;
            Pdef.single_batch_inputs = {};
            Pdef.inputs  = [];
            Pdef.outputs = [];
            Pdef.backward_input_diff = true;
            
            PARAM = xmerge_struct( Pdef, PARAM );
            obj.param = PARAM;
            
            if isa(solver_or_net, 'function_handle') 
                % use loader func to construct with PARAM
                solver_or_net_load_func = solver_or_net;
                LOADER_PARAM = PARAM;
                % use inf to indicate artificial loss_weight for force backpropagation
                LOADER_PARAM.dummy_output_loss_weight = inf;
                solver_or_net = solver_or_net_load_func( LOADER_PARAM );
            end
            
            obj.worker = solver_or_net;
            
            switch class(obj.worker)
                case 'caffe.Solver'
                    obj.is_solver = 1;
                    obj.net = obj.worker.net;
                    obj.solver = obj.worker;
                    obj.func_forward  = @() obj.worker.net.forward_prefilled();
                    obj.func_backward = @() obj.worker.net.backward_prefilled();
                    obj.func_update   = @() obj.worker.update();
                case 'caffe.Net'
                    obj.is_solver = 0;
                    obj.net = obj.worker;
                    obj.func_forward  = @() obj.worker.forward_prefilled();
                    obj.func_backward = @() obj.worker.backward_prefilled();
                case 'matcaffeSolver'
                    obj.is_solver = 1;
                    obj.net = obj.worker.net;
                    obj.solver = obj.worker.solver;
                    obj.func_forward  = @() obj.worker.forward();
                    obj.func_backward = @() obj.worker.backward();
                    obj.func_update   = @() obj.worker.update();
                case 'matcaffeNet'
                    obj.is_solver = 0;
                    obj.net = obj.worker.net;
                    obj.func_forward  = @() obj.worker.forward();
                    obj.func_backward = @() obj.worker.backward();
                otherwise
                    error( 'Unrecognized solver or net' );
            end
            
            if obj.is_solver
                % pre-run update to avoid further problems
                obj.func_update();
            end
            
            obj.input_subbatch_idx  = 0;
            obj.output_subbatch_idx = 0;
            obj.num_subbatch        = 0;
            obj.last_subbatch_ratio  = 0;
            
            obj.rand_stream = RandStream('mrg32k3a','seed',82943);
            
            % inputs
            if iscell( PARAM.inputs )
                assert( all(ismember(PARAM.inputs, obj.net.blob_names)), ...
                    'unknown input blob names' );
                obj.input_names = vec(PARAM.inputs);
            else
                obj.input_names = obj.net.inputs;
            end
            if isscalar(obj.param.backward_input_diff)
                obj.backward_input_diff = repmat( ...
                    boolean( obj.param.backward_input_diff ), ...
                    size(obj.input_names) );
            else
                obj.backward_input_diff = reshape( ...
                    boolean( obj.param.backward_input_diff ), ...
                    size(obj.input_names) );
            end
            
            % single_batch_inputs
            assert( iscell( obj.param.single_batch_inputs ), ...
                'param.single_batch_inputs must a cell array' );
            for k = 1:numel(obj.param.single_batch_inputs)
                sbik = obj.param.single_batch_inputs{k};
                if isnumeric(sbik)
                    obj.param.single_batch_inputs{k} = obj.input_names{ sbik };
                else
                    assert( ischar(sbik), ...
                        'element of param.single_batch_inputs should be either numeric index or blob name' );
                end
            end
            
            % outputs
            if iscell( PARAM.outputs )
                assert( all(ismember(PARAM.outputs, obj.net.blob_names)), ...
                    'unknown output blob names' );
                obj.output_names = vec(PARAM.outputs);
            else
                obj.output_names = obj.net.outputs;
            end
            obj.actual_output_idxb = ismember(obj.output_names,obj.net.outputs);
            
            % batch sizes
            for k = 1:numel(obj.input_names)
                [obj.input_max_batch_sizes(k), obj.input_batch_dims(k)] = ...
                    obj.BlobBatchSize( obj.input_names{k} );
            end
            obj.input_max_batch_sizes = obj.input_max_batch_sizes * obj.param.batch_size_mult;
            
            for k = 1:numel(obj.output_names)
                [obj.output_max_batch_sizes(k), obj.output_batch_dims(k)] = ...
                    obj.BlobBatchSize( obj.output_names{k} );
            end
            obj.output_max_batch_sizes = obj.output_max_batch_sizes * obj.param.batch_size_mult;
            
            % loss weights
            obj.blob_loss_weights = obj.net.blob_loss_weights;
            dummy_loss_weight_idx = find( isinf(obj.blob_loss_weights) );
            for k = vec(dummy_loss_weight_idx).' % clear dummy loss weight
                B = obj.net.blob_vec(k);
                bsz = B.shape();
                if isempty(bsz)
                    bsz = size(B.get_data());
                end
                bsz = canonical_size_vec(bsz,2);
                B.set_data( zeros(bsz,'single') );
            end
            obj.blob_loss_weights(isinf(obj.blob_loss_weights)) = 0;
            obj.loss_output_idxb = ismember( obj.output_names, ...
                obj.net.blob_names(obj.blob_loss_weights~=0) );
            
            % input output refresh tag
            obj.input_data_subbatch_id  = zeros(size(obj.input_names));
            obj.output_diff_subbatch_id = zeros(size(obj.output_names));
            
            % base iter
            obj.base_iter = 0;
            
        end
        
        function UseDeviceMat( obj, use_device )
            if nargin<2, use_device = 1; end
            obj.use_device = use_device;
        end
        
        function UseCPUMat( obj )
            obj.UseDeviceMat(0);
        end
        
        function A = todev( obj, A )
            if obj.use_device
                if ~isa(A,'gpuArray')
                    A = gpuArray(A);
                end
            end
        end
        
        function rng( obj,rand_seed )
            if nargin<2 
                obj.rand_stream = RandStream('mrg32k3a');
            else
                obj.rand_stream = RandStream('mrg32k3a','seed',rand_seed);
            end
        end
        
        function varargout = GetInputDiff( obj, varargin )
            varargout = obj.input_diff;
        end
        
        function varargout = GetOutputData( obj, varargin )
            varargout = obj.output_data;
        end
        
        function SetInputData( obj, varargin )
            % SetInputData( Blob1, Blob2, ... )
            % SetInputData( Name1, Blob1, Name2, Blob2, ... )
            if isempty(obj.input_data)
                I = cell( numel(obj.input_names),1 );
                U = I;
                obj.input_data  = I;
                obj.input_index = U;
                obj.num_subbatch = 0;
            else
                I = obj.input_data;
                U = obj.input_index;
            end
            updated_idxb = false( size(I) );
            if isempty( varargin ), return; end
            if ischar( varargin{1} )
                namedargin = reshape(varargin,2,numel(varargin)/2);
                for j = 1:size(namedargin,2)
                    [c, k] = ismember( namedargin{1,j}, obj.input_names );
                    assert( c, 'no such input blob' );
                    I{k} = namedargin{2,j};
                    updated_idxb(k) = true;
                end
            else
                assert( numel(varargin) == numel(obj.input_names), ...
                    'input number mismatched' );
                I(:) = varargin;
                updated_idxb(:) = true;
            end
            for k = find(vec(updated_idxb).')
                if iscell(I{k})
                    assert( numel(I{k})==2, 'wrong input format' );
                    U{k} = vec(I{k}{2});
                    I{k} = I{k}{1};
                else
                    U{k} = (1:size(I{k},abs(obj.input_batch_dims(k)))).';
                end
            end
            obj.input_subbatch_idx = 0;
            obj.input_data  = I;
            obj.input_index = U;
            obj.input_data_subbatch_id(updated_idxb) = 0;
            
            obj.input_info = repmat( struct( 'size', [], 'ref_type', [] ), ...
                size(obj.input_data) ) ;
            for k = 1:numel(obj.input_data)
                obj.input_info(k).size     = size(obj.input_data{k});
                if obj.use_device
                    obj.input_info(k).ref_type = gpuArray(single([]));
                else
                    obj.input_info(k).ref_type = zeros(0,0,'like',obj.input_data{k});
                end
            end
        end
        
        function SetOutputDiff( obj, varargin )
            % SetOutputDiff( Blob1, Blob2, ... )
            % SetoutputDiff( Name1, Blob1, Name2, Blob2, ... )
            if isempty(obj.output_diff)
                I = cell( numel(obj.output_names), 1 );
                obj.output_diff = I;
            else
                I = obj.output_diff;
            end
            I(obj.loss_output_idxb) = {[]};
            if isempty( varargin ), return; end
            
            updated_idxb = false( size(I) );
            if ischar( varargin{1} )
                namedargin = reshape(varargin,2,numel(varargin)/2);
                for j = 1:size(namedargin,2)
                    [c, k] = ismember( namedargin{1,j}, obj.output_names );
                    assert( c, 'no such output blob' );
                    obj.check_single_output_diff( k, namedargin{2,j} );
                    I{k} = namedargin{2,j};
                end
            else
                assert( numel(varargin) == numel(obj.output_names), ...
                    'output number mismatched' );
                for k = 1:numel(obj.output_names)
                    obj.check_single_output_diff( k, varargin{k} );
                end
                I(:) = varargin;
                updated_idxb(:) = true;
            end
            
            assert( all( cellfun( @isempty, I(obj.loss_output_idxb) ) ), ...
                'Should not set diff for loss outputs. It is set automatically' );
            
            obj.output_subbatch_idx = 0;
            obj.output_diff = I;
            obj.output_diff_subbatch_id(updated_idxb | obj.loss_output_idxb) = 0;
        end
        
        function [batch_size, batch_dim] = BlobBatchSize( obj, blob_name )
            B = obj.net.blobs(blob_name);
            s = B.shape();
            if isempty(s)
                batch_size = numel(B.get_data());
                batch_dim  = -1;
            else
                batch_size = s(end);
                batch_dim  = length(s);
                if ismember( blob_name, obj.param.single_batch_inputs )
                    batch_dim = -batch_dim;
                end
            end
        end
        
        function Forward( obj )
            obj.pre_forward();
            for i = 1:obj.num_subbatch
                obj.forward_subbatch(i);
            end
            obj.post_forward();
            if obj.num_subbatch==1 || ~obj.is_solver
                if ~isempty(obj.input_data)
                    obj.input_data(:)={[]};
                end
            end
        end
        
        function Backward( obj )
            obj.pre_backward();
            for i = obj.num_subbatch:-1:1 % reverse order to reduce forward pass
                if obj.input_subbatch_idx ~= i
                    obj.forward_subbatch(i,false);
                end
                obj.backward_subbatch(i);
            end
            obj.post_backward();
            if ~isempty(obj.output_diff),
                obj.output_diff(:)={[]};
            end
            if ~isempty(obj.input_data)
                obj.input_data(:)={[]};
            end
        end
                
        function Update( obj )
            if obj.memory_econ,
                if ~isempty( obj.output_data )
                    obj.output_data(:)={[]};
                end
                if ~isempty( obj.input_diff )
                    obj.input_diff(:)={[]};
                end
            end
            obj.func_update();
        end
        
        function ForwardBackward( obj )
            obj.pre_forward();
            obj.pre_backward();
            for i = 1:obj.num_subbatch
                obj.forward_subbatch(i);
                obj.backward_subbatch(i);
            end
            obj.post_forward();
            obj.post_backward();
            if obj.memory_econ,
                if ~isempty(obj.output_diff),
                    obj.output_diff(:)={[]};
                end
                if ~isempty(obj.input_data)
                    obj.input_data(:)={[]};
                end
            end
        end
        
        function Save( obj, caffemodel_path )
            mkpdir_p(caffemodel_path);
            [pdir,fn,ext] = fileparts(caffemodel_path);
            if ismember(ext, {'.caffemodel','.solverstate'})
                caffemodel_path = fullfile(pdir,fn);
            end
            if obj.is_solver
                obj.solver.snapshot( caffemodel_path );
                S = struct();
                S.start_iteration = obj.base_iter;
                S.iteration   = obj.solver.iter() + S.start_iteration;
                S.solver_type = obj.solver.type();
                struct2prototxt( S, [caffemodel_path '.caffemodel' '.ss-pt'] );
            else
                caffemodel_path1 = [caffemodel_path '.caffemodel'];
                obj.net.save( caffemodel_path1 );
            end
            try 
                STAT = obj.statistics();
                save_from_struct( [caffemodel_path '.statistics.mat'], STAT );
                writetextfile( [display_struct(STAT) sprintf('\n')], ...
                    [caffemodel_path '.statistics.txt'] );
            catch
            end
        end
        
        function Load( obj, caffemodel_path )
            obj.TryLoad( caffemodel_path, true );
        end

        function TryLoad_Solver( obj, caffemodel_path, no_try )
            if nargin<3 || isempty(no_try)
                no_try = false;
            end
            obj.TryLoad( caffemodel_path, no_try, 1 );
        end
        
        function TryLoad( obj, caffemodel_path, no_try, try_to_load_solverstate )
            if nargin<3 || isempty(no_try)
                no_try = false;
            end
            if nargin<4 || isempty(try_to_load_solverstate)
                try_to_load_solverstate = 0;
            end
            [pdir,fn,ext] = fileparts(caffemodel_path);
            if strcmp(ext,'.solverstate')
                caffemodel_path = fullfile(pdir,[fn '.caffemodel']);
                [pdir,fn,ext] = fileparts(caffemodel_path);
            end
            file_existed = boolean(exist(caffemodel_path,'file'));
            if exist( caffemodel_path, 'file' )
                caffemodel_path1 = caffemodel_path;
            elseif ~strcmp(ext, '.caffemodel')
                caffemodel_path1 = [caffemodel_path '.caffemodel'];
                file_existed = boolean(exist(caffemodel_path1,'file'));
            end
            if ~file_existed
                if no_try
                    error( 'model does not exist' );
                else
                    return;
                end
            end
            [pdir,fn,ext] = fileparts(caffemodel_path1);
            caffemodel_path0 = fullfile(pdir,fn);
            if try_to_load_solverstate && obj.is_solver
                solver_fn = [caffemodel_path0 '.caffemodel' '.ss-pt'];
                S = struct();
                if exist( solver_fn, 'file' )
                    S = prototxt2struct( solver_fn );
                end
                cached_solver_type = default_eval( 'S.solver_type', [] );
                caffe_solverstate_path = [caffemodel_path0 '.solverstate'];
                
                to_load_solver = 1;
                if isempty(cached_solver_type)
                    to_load_solver = 0;
                    fprintf('No solver type was recorded. ');
                elseif ~strcmp(cached_solver_type, obj.solver.type())
                    fprintf('Solver type is not the same as the current solver. ');
                elseif ~exist( caffe_solverstate_path, 'file' )
                    fprintf('Cannot find the solverstate file. ');
                end
                if ~to_load_solver
                    fprintf('Do not load solverstate. Only load caffemodel\n');
                end
                if to_load_solver
                    obj.solver.restore(caffe_solverstate_path);
                else
                    obj.net.copy_from( caffemodel_path1 );
                end
                
                total_iteration = default_eval( 'S.iteration', 0 ); 
                obj.base_iter = default_eval( 'S.start_iteration', 0 );
                obj.solver.set_iter( total_iteration - obj.base_iter );
            else
                obj.net.copy_from( caffemodel_path1 );
            end
        end
        
    end
    
    % utils
    methods (Access = protected)
        function check_single_output_diff( obj, k, D )
            internal_shape = obj.net.blobs( obj.output_names{k} ).shape();
            internal_shape = internal_shape(1:(end-1));
            if iscell(D)
                assert( numel(D)==2, 'Wrong output diff format' );
                D = D{1};
            end
            external_shape = canonical_size_vec( size(D), length(internal_shape) );
            external_shape = external_shape(1:length(internal_shape));
            assert( all( internal_shape == external_shape ), ...
                'output shape mismatched' );
        end
        function canonicalize_output_diff( obj )
            Darr = obj.output_diff;
            if ~iscell(Darr)
                Darr = cell(1,numel(obj.output_names));
            end
            for k = 1:numel(Darr)
                if iscell( Darr{k} )
                    I = Darr{k}{1};
                    U = Darr{k}{2};
                    batch_dim = obj.output_batch_dims(k);
                    blob_shape = canonical_size_vec( size(obj.output_data{k}), batch_dim );
                    D = zeros( blob_shape, 'like', I );
                    uniqU = unique(U);
                    assert( ~any(uniqU>blob_shape(batch_dim)) && ...
                        ~any(uniqU<1), 'output diff index out of range' );
                    ref2prebatch_dim = repmat({':'},1,batch_dim-1);
                    for j = vec(uniqU).'
                        j_idxb = (U==j);
                        s1 = substruct('()',[ref2prebatch_dim,{j_idxb}]);
                        D_j = sum( subsref( I, s1 ), batch_dim ); % sum diff
                        s2 = substruct('()',[ref2prebatch_dim,{j}]);
                        D = subsasgn( D, s2, D_j );
                    end
                    Darr{k} = D;
                end
            end
            obj.output_diff = Darr;
        end
    end
    
    % Forward
    methods (Access = protected)
        
        function pre_forward( obj )
            % figure out subbatch
            obj.num_subbatch = 1;
            obj.last_subbatch_ratio = 1;
            is_first_blob = 1;
            for k = 1:numel(obj.input_names)
                [internal_batch_size, batch_dim] = deal( ...
                    obj.input_max_batch_sizes(k), obj.input_batch_dims(k) );
                
                U = obj.input_index{k};
                if batch_dim < 0,
                    assert( internal_batch_size == numel(U), ...
                        'non-batch numel does not match' );
                    continue;
                end
                
                external_batch_size = numel(U);
                
                cur_num_subbatch = ceil(external_batch_size/internal_batch_size);
                cur_last_subbatch_ratio = external_batch_size / internal_batch_size - ...
                    (cur_num_subbatch-1);
                if is_first_blob
                    obj.num_subbatch = cur_num_subbatch;
                    obj.last_subbatch_ratio = cur_last_subbatch_ratio;
                    is_first_blob = 0;
                else
                    assert( obj.num_subbatch == cur_num_subbatch, 'subbatch total mismatched' );
                    assert( cur_last_subbatch_ratio == obj.last_subbatch_ratio, ...
                        'subbatch element mismatched' );
                end
            end
            obj.subbatch_ratios = zeros(1,obj.num_subbatch);
            obj.subbatch_ratios(1:(end-1)) = 1/(obj.num_subbatch-1+obj.last_subbatch_ratio);
            obj.subbatch_ratios(end) = obj.last_subbatch_ratio / ...
                (obj.num_subbatch-1+obj.last_subbatch_ratio);
            % set up output variable
            obj.output_data = cell(1,numel(obj.output_names));
            for k = 1:numel(obj.output_data) 
                batch_dim = obj.output_batch_dims(k);
                if batch_dim < 0
                    obj.output_data{k} = cell(1);
                else
                    obj.output_data{k} = cell( canonical_size_vec( ...
                        [ones(1,batch_dim-1),obj.num_subbatch], 2) );
                end
            end
        end
        
        function forward_subbatch( obj, i, update_output )
            if nargin<3
                update_output = true;
            end
            % set inputs
            num_input = numel(obj.input_names);
            subbatch_input_blobs = cell(1,num_input);
            cur_subbatch_ids = repmat(i,1,num_input);
            for k = 1:num_input
                [internal_batch_size, batch_dim] = deal( ...
                    obj.input_max_batch_sizes(k), obj.input_batch_dims(k) );
                I = obj.input_data{k};
                U = obj.input_index{k};
                if batch_dim < 0
                    P  = subsref(I,substruct('()',[repmat({':'},1,abs(batch_dim)-1), {U}]));
                    cur_subbatch_ids(k) = 1;
                else
                    external_batch_size = numel(U);
                    st = (i-1)*internal_batch_size;
                    en = min( st+internal_batch_size, external_batch_size );
                    st = st+1;
                    P = subsref(I,substruct('()',[repmat({':'},1,batch_dim-1), {U(st:en)}]));
                end
                P = single(P);
                subbatch_input_blobs{k} = P;
            end
            input_blob_shapes = cell(1,num_input);
            for k = 1:num_input
                input_blob_shapes{k} = canonical_size_vec( ...
                    size( subbatch_input_blobs{k} ), ...
                    abs( obj.input_batch_dims(k) ) ...
                    );
            end
            obj.net.reshape_input_blobs(input_blob_shapes);
            for k = 1:num_input
                if cur_subbatch_ids(k)==obj.input_data_subbatch_id(k), continue; end
                blob_name = obj.input_names{k};
                B = obj.net.blobs( blob_name );
                B.set_data( subbatch_input_blobs{k} );
                obj.input_data_subbatch_id(k) = cur_subbatch_ids(k);
            end
            % do forward
            obj.func_forward();
            % gather output data
            if update_output
                subbatch_ratio = obj.todev( obj.subbatch_ratios(i) );
                for k = 1:numel(obj.output_names)
                    B = obj.net.blobs( obj.output_names{k} );
                    if obj.use_device
                        P = B.get_device_data();
                    else
                        P = B.get_data();
                    end
                    batch_dim = obj.output_batch_dims(k);
                    if batch_dim < 0
                        if isempty(obj.output_data{k}{1})
                            obj.output_data{k}{1} = P*subbatch_ratio;
                        else
                            obj.output_data{k}{1} = ...
                                obj.output_data{k}{1} + P*subbatch_ratio;
                        end
                    else
                        obj.output_data{k}{i} = P;
                    end
                end
            end
            % update status
            obj.input_subbatch_idx = i;
        end
        
        function post_forward( obj )
            if obj.use_device
                for i = 1:numel(obj.output_data)
                    batch_dim = abs(obj.output_batch_dims(i));
                    obj.output_data{i} = cat(batch_dim, obj.output_data{i}{:});
                end
            else
                for i = 1:numel(obj.output_data)
                    obj.output_data{i} = cell2mat(obj.output_data{i});
                end
            end
        end
        
    end
    
    % Backward
    methods (Access = protected)
        
        function pre_backward( obj )
            % figure out subbatch
            obj.canonicalize_output_diff(); % if index, then to normal map
            for k = 1:numel(obj.output_names)
                
                if obj.loss_output_idxb(k)
                    assert( isempty(obj.output_diff{k}), ...
                        'diff for loss blob should not be set' );
                    continue;
                end
                
                [internal_batch_size, batch_dim] = deal( ...
                    obj.output_max_batch_sizes(k), obj.output_batch_dims(k) );
                                
                if batch_dim < 0,
                    assert( isempty(obj.output_diff{k}) || ... % empty blob will be handled in backward_subbatch
                        internal_batch_size == size(obj.output_diff{k}, -batch_dim), ...
                        'non-batch numel does not match' );
                    continue;
                end
                if isempty(obj.output_diff{k}),
                    continue;
                end
                
                I = obj.output_diff{k};
                external_shape = canonical_size_vec( size(I), batch_dim );
                external_batch_size = external_shape( batch_dim );
                
                cur_num_subbatch = ceil(external_batch_size/internal_batch_size);
                cur_last_subbatch_ratio = external_batch_size / internal_batch_size - ...
                    (cur_num_subbatch-1);

                assert( obj.num_subbatch == cur_num_subbatch, 'subbatch total mismatched' );
                assert( cur_last_subbatch_ratio == obj.last_subbatch_ratio , ...
                    'subbatch element mismatched' );
                
            end
            % set up input variable
            obj.input_diff = cell(1,numel(obj.input_names));
            for k = 1:numel(obj.input_diff)
                if ~obj.need_backward_input_diff(k)
                    continue;
                end
                obj.input_diff{k} = zeros( obj.input_info(k).size, ...
                    'like', obj.input_info(k).ref_type );
            end
        end
        
        function backward_subbatch( obj, i, update_input )
            if nargin<3
                update_input = true;
            end
            
            % set outputs
            num_output = numel(obj.output_names);
            subbatch_output_blobs = cell(1,num_output);
            cur_subbatch_ids = repmat(i,1,num_output);
            for k = 1:num_output
                if obj.loss_output_idxb(k), continue; end
                [internal_batch_size, batch_dim] = deal( ...
                    obj.output_max_batch_sizes(k), obj.output_batch_dims(k) );
                if isempty(obj.output_diff{k})
                    P = [];
                else
                    if batch_dim < 0
                        P = obj.output_diff{k};
                        cur_subbatch_ids(k) = 1;
                    else
                        I  = obj.output_diff{k};
                        external_batch_size = size(I,batch_dim);
                        st = (i-1)*internal_batch_size;
                        en = min( st+internal_batch_size, external_batch_size );
                        st = st+1;
                        P  = subsref(I,substruct('()',[repmat({':'},1,batch_dim-1), {st:en}]));
                    end
                    P = single(P);
                end
                subbatch_output_blobs{k} = P;
            end
            
            subbatch_ratio = obj.subbatch_ratios(i);
            for k = vec(find(obj.loss_output_idxb)).'
                if cur_subbatch_ids(k)==obj.output_diff_subbatch_id(k), 
                    continue;
                end
                blob_name = obj.output_names{k};
                [~,blob_id] = ismember( blob_name, obj.net.blob_names );
                B = obj.net.blob_vec(blob_id);
                bsz = B.shape();
                if isempty(bsz)
                    bsz = size(B.get_data());
                end
                bsz = canonical_size_vec(bsz,2);
                subbatch_loss_weight = double(subbatch_ratio) * ...
                    double( obj.blob_loss_weights(blob_id) );
                subbatch_output_blobs{k} = repmat( single(subbatch_loss_weight), bsz );
                cur_subbatch_ids(k) = subbatch_ratio; % subbatch_ratio as cache id
            end
            % obj.net.reshape_as_output(subbatch_output_blobs); % invalid
            for k = 1:num_output
                if ~obj.actual_output_idxb(k),
                    continue;  % skip none actual outoput blobs (try to use split to enable that before load the net)
                end
                if cur_subbatch_ids(k)==obj.output_diff_subbatch_id(k), continue; end
                blob_name = obj.output_names{k};
                B = obj.net.blobs( blob_name );
                if isempty(subbatch_output_blobs{k} )
                    if isempty(B.shape)
                        if obj.use_device
                            diffB = B.get_device_diff();
                        else
                            diffB = B.get_diff();
                        end
                        diffB(:) = 0;
                        B.set_diff(diffB);
                    else
                        szB = canonical_size_vec( B.shape, 2 );
                        B.set_diff( zeros(szB,'single') );
                    end
                else
                    B.set_diff( subbatch_output_blobs{k} );
                end
                obj.output_diff_subbatch_id(k) = cur_subbatch_ids(k);
            end
            % do backward
            obj.func_backward();
            % gather input diff
            if update_input
                for k = 1:numel(obj.input_names)
                    if ~obj.need_backward_input_diff(k)
                        continue;
                    end
                    B = obj.net.blobs( obj.input_names{k} );
                    if obj.use_device
                        P = B.get_device_diff();
                    else
                        P = B.get_diff();
                    end
                    
                    [internal_batch_size, batch_dim] = deal( ...
                        obj.input_max_batch_sizes(k), obj.input_batch_dims(k) );
                    U = obj.input_index{k};
                    if batch_dim < 0
                        bU = U;
                    else
                        external_batch_size = numel(U);
                        st = (i-1)*internal_batch_size;
                        en = min( st+internal_batch_size, external_batch_size );
                        st = st+1;
                        bU = U(st:en);
                    end
                    
                    % accum index
                    [ubU, ~, ubUic] = unique( uint64(bU) );
                    if numel(ubU) == numel(bU)
                        if obj.use_device
                            bU = gpuArray(bU);
                        end
                        ss = substruct('()',[repmat({':'},1,abs(batch_dim)-1), {bU}]);
                        obj.input_diff{k}  = subsasgn( obj.input_diff{k}, ss, ...
                            subsref( obj.input_diff{k}, ss ) + P );
                    else
                        batch_size = size(P,abs(batch_dim));
                        szP = canonical_size_vec(size(P),abs(batch_dim));
                        P = reshape( P, numel(P)/batch_size, batch_size );
                        
%                         t1=tic_print('naive: ');
%                         Q = zeros( size(P,1), numel(ubU), 'like', P );
%                         for j=1:numel(ubU)
%                             cidxb  = (bU==j);
%                             Q(:,j) = sum(P(:,cidxb),2);
%                         end
%                         toc_print(t1);
                        
                        sU = uint64(1:size(P,1));
                        if obj.use_device
                            sU = gpuArray(sU);
                            ubUic = gpuArray(uint64(ubUic));
                        end
                        mUArr = zeros( [size(P), 2], 'like', sU );
                        mUArr(:,:,1) = repmat( vec(sU), 1, size(P,2) );
                        mUArr(:,:,2) = repmat( vec(ubUic).', size(P,1), 1 );
                        Q = accumarray( reshape( mUArr, numel(mUArr)/2, 2 ), P(:), [size(P,1), numel(ubU)] );
                        
                        szQ = szP; szQ(abs(batch_dim)) = numel(ubU);
                        Q = reshape( Q, szQ );
                        ss = substruct('()',[repmat({':'},1,abs(batch_dim)-1), {ubU}]);
                        obj.input_diff{k}  = subsasgn( obj.input_diff{k}, ss, ...
                            subsref( obj.input_diff{k}, ss ) + Q );
                    end
                    
                end
            end
            % update status
            obj.output_subbatch_idx = i;
        end
        
        function post_backward( obj )
            % do nothing here
        end
        
    end
    
    methods ( Access=public )
        function c = need_backward_input_diff( obj, input_id )
            c = obj.backward_input_diff(input_id);
        end
        function set_backward_input_diff( obj, input_id, need_backward )
            assert(input_id<=numel(obj.backward_input_diff), ...
                'input blob id out of range' );
            obj.backward_input_diff(input_id) = boolean(need_backward);
        end
        
        function SetMemoryEcon( obj, use_memory_econ )
            obj.memory_econ = boolean(use_memory_econ);
        end
        
        function ToggleInputDiff( obj, input_id, need_backward )
            obj.set_backward_input_diff(input_id, need_backward);
        end
        
        function A = statistics( obj )
            A = matcaffe_net_statistics( obj.net );
        end
    end
    
end
