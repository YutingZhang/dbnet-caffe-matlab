classdef matcaffeNet < handle
    properties ( Access = public )
        net
        input
        tag
    end
    properties ( Access = private )
        is_net_holder
        run_iters
        is_shared_
    end
    methods  ( Access = public )
        function obj = matcaffeNet( net, is_net_holder )
            if nargin<2, obj.is_net_holder = false;
            else obj.is_net_holder = is_net_holder; end
            obj.net    = net;
            obj.input  = struct('loader',{},'data',{},'label', {});
            obj.is_shared_ = 0;
        end
        function delete( obj )
            if obj.is_net_holder
                obj.net.reset();
            end
        end
        function copy_from( obj, src_net )
            obj.net.copy_from(src_net);
        end
        function share_weights_with( obj, src_net )
            if isa( src_net, 'matcaffeNet' )
                src_caffe_net = src_net.net;
            else
                src_caffe_net = src_net;
            end
            obj.netnet_share_trained_layers_with( src_caffe_net );
            obj.is_shared_ = 1;
        end
        function s = is_shared( obj )
            s = obj.is_shared_;
        end
        function add_loader( obj, loader_obj, data_blob_names, label_name )
            
            if isempty(obj.input)
                exB = {};
            else
                exB1 = cat(1,obj.input.data);
                exB2 = cat(1,obj.input.label);
                exB2(~cellfun(@ischar,{exB2.name}.')) = [];
                exB = [{exB1.name},{exB2.name}];
            end
            
            I = empty_struct(obj.input);
            I.loader = loader_obj;
            if ischar(data_blob_names), data_blob_names = {data_blob_names}; end
            data_blob_names = vec(data_blob_names);
            used_idx = find( ~cellfun(@isempty,data_blob_names) );
            assert( all(ismember(data_blob_names(used_idx),obj.net.inputs)), 'data not input blob' );
            assert( all(~ismember(data_blob_names(used_idx),exB)), 'duplicate assement of data to input blobs' );
            I.data = struct( 'name', data_blob_names(used_idx), ...
                'output_id', num2cell(used_idx), ...
                'blob', cellfun(@(a) obj.net.blobs(a),data_blob_names(used_idx),'UniformOutput',0) );
            if exist('label_name','var') && ischar(label_name)
                assert( ismember(label_name,obj.net.inputs), 'label not input blob' );
                assert( ~ismember(label_name,exB), 'duplicate assement of label to input blobs' );
                I.label = struct( 'name', label_name, 'blob', obj.net.blobs(label_name) );
            else
                I.label = struct( 'name', [], 'blob', [] );
            end
            
            obj.input(end+1) = I;
        end
        function fill_data( obj )
            for j = 1:numel(obj.input)
                [D,lb] = obj.input(j).loader.get_batch;
                nb = numel( obj.input(j).data );
                for i = 1:nb
                    B = obj.input(j).data(i).blob;
                    B.set_data( D{ obj.input(j).data(i).output_id } );
                end
                if ischar(obj.input(j).label.name)
                    B = obj.input(j).label.blob;
                    B.set_data( reshape(single(lb),B.shape) );
                end
            end
        end
        function forward(obj)
            obj.fill_data();
            obj.net.forward_prefilled();
        end
        function backward(obj)
            obj.net.backward_prefilled();
        end
        function set_run_iters( obj, run_iters )
            obj.run_iters = run_iters;
        end
        function varargout = run( obj, output_blob_names, ave_func )
            if nargin<2
                output_blob_names = obj.net.outputs;
            end
            output_blob_names = vec(output_blob_names).';
            numOutput = numel(output_blob_names);
            if nargin<3
                ave_func = @mean;
            end
            if ~iscell( ave_func ), ave_func = {ave_func}; end
            if isscalar(ave_func), ave_func = repmat( ave_func, 1, numOutput ); end
            blobOutput = cell(1,numOutput);
            aveOutput  = cell(1,numOutput);
            shapeOutput = cell(1,numOutput);
            for j = 1:numOutput
                blobOutput{j} = obj.net.blobs(output_blob_names{j});
                s = blobOutput{j}.shape();
                if isempty(s)
                    s = [1 1];
                end
                shapeOutput{j} = s;
                aveOutput{j}  = zeros( [prod(s), obj.run_iters], 'like', blobOutput{j}.get_data() );
            end
            blobOutput = cat(2,blobOutput{:});
            for k = 1:obj.run_iters
                obj.fill_data();
                obj.net.forward_prefilled();
                for j = 1:numOutput
                    aveOutput{j}(:,k) = vec( ave_func{j}( blobOutput(j).get_data(), numel(shapeOutput{j}) ) );
                end
            end
            for j = 1:numOutput
                aveOutput{j} = ave_func{j}( aveOutput{j}, 2 );
                aveOutput{j} = reshape( aveOutput{j}, canonical_size_vec( shapeOutput{j}, 2 ) );
            end
            varargout = aveOutput;
        end
        function outputS = run4struct( obj, output_blob_names, varargin )
            if nargin<2
                output_blob_names = obj.net.outputs;
            end
            outputA = cell( 2, numel(output_blob_names) );
            outputA(1,:) = cellfun( @str2valid_varname, vec(output_blob_names).', 'UniformOutput', false );
            [outputA{2,:}] = obj.run( output_blob_names );
            outputS = struct( outputA{:} );
        end
        function S = run4stat( obj, stat_func )
            if ischar(stat_func)
                switch stat_func
                    case 'sparsity'
                        stat_func = @(a) sum( abs(a(:))<gpuArray(eps('single')) )./gpuArray(single(numel(a)));
                    otherwise
                        error('Unrecognized stat_func');
                end
            else
                assert( isa( stat_func, 'function_handle' ) );
            end
            blob_names = obj.net.blob_names;
            blob_num = length(blob_names);
            S = repmat( struct('name',[],'val',{cell(obj.run_iters,1)}), blob_num, 1 );
            for k = 1:obj.run_iters
                obj.fill_data();
                obj.net.forward_prefilled();
                for j = 1:blob_num
                    S(j).name = blob_names{j};
                    D = obj.net.blobs(blob_names{j}).get_device_data();
                    S(j).val{k}  = gather( stat_func(D) );
                end
            end
            for j = 1:blob_num
                nd = ndims( S(j).val{1} );
                S(j).val = mean( cat(nd+1, S(j).val{:}), nd+1 );
            end
        end
        function A = statistics( obj )
            A = matcaffe_net_statistics( obj.net );
        end
    end
end
