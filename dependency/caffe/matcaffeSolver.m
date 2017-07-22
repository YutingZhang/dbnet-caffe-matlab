classdef matcaffeSolver < matcaffeNet
    properties ( Access = public )
        solver
        test_nets
    end
    properties ( Access = private )
        is_solver_holder
    end
    methods  ( Access = public )
        function obj = matcaffeSolver( solver, is_solver_holder )
            obj@matcaffeNet( solver.net, false );
            if nargin<2, obj.is_solver_holder = false;
            else obj.is_solver_holder = is_solver_holder; end
            obj.solver = solver;
        end
        function delete( obj )
            if obj.is_solver_holder
                obj.solver.reset();
            end
        end
        function step( obj, num_steps )
            if nargin<2
                num_steps = 1;
            end
            k = 0;
            while k<num_steps && obj.solver.iter<obj.solver.max_iter
                k = k+1;
                obj.fill_data( );
                obj.solver.step(1);
            end
        end
        function update(obj)
            obj.solver.update();
        end
        function add_test_net( obj, net, test_iter, share_weights )
            if ~isa( net, 'matcaffeNet' )
                assert( isa(net, 'caffe.Net'), 'wrong type' );
                caffe_net = net;
                net = matcaffeNet(caffe_net,false);
            end
            if ~exist( 'share_weights', 'var' ) || isempty(share_weights)
                share_weights = false;
            end
            net.set_run_iters( test_iter );
            if share_weights
                assert( isempty( default_eval('net.tag.linked_solver',[]) ), 'network has been linked with another solver' );
                net.share_weights_with(obj.net);
                net.tag.linked_solver = obj.solver.get_handle;
            end
            obj.test_nets = [obj.test_nets;net];
        end
        function sync_test_net( obj, ids )
            if nargin<2
                ids = 1:numel(obj.test_nets);
            elseif islogical(ids)
                ids = find(ids);
            end
            ids = vec(ids).';
            assert( all( numel(obj.test_nets) >= ids ), 'no such test_net' );
            for k = ids
                net = obj.test_nets(k);
                if ~isequal( default_eval('net.tag.linked_solver',[]), obj.solver.get_handle() )
                    obj.test_nets(k).net.copy_from(obj.solver.net);
                end
            end
        end
        function R = run_test( obj, ids )
            if nargin<2
                ids = 1:numel(obj.test_nets);
            elseif islogical(ids)
                ids = find(ids);
            end
            ids = vec(ids).';
            obj.sync_test_net( ids );
            R = cell(numel(obj.test_nets),1);
            for k = ids
                R{k} = obj.test_nets(k).run4struct();
            end
        end
        function R = run_single_test( obj, id )
            if nargin<2
                id = 1;
            end
            if id==0
                R = obj.run4struct();
            else
                obj.sync_test_net( id );
                R = obj.test_nets(id).run4struct();
            end
        end
        function restore( obj, iter_or_fn )
            if nargin<2, iter_or_fn = inf; end
            sp = obj.solver.snapshot_prefix;
            if ischar(iter_or_fn)
                fn = iter_or_fn;
                assert( ~isempty(fn), 'filename should not be empty' );
            else
                fn_func = @(iter_str) [sp '_iter_' iter_str '.solverstate'];
                iter = iter_or_fn;
                if isinf(iter)
                    ss_list = dir( fn_func('*') );
                    ss_list = { ss_list.name };
                    iter = -1;
                    for k = 1:length(ss_list)
                        iter_str=regexp( ss_list{k}, '_([0-9]*)\.solverstate$', 'tokens' );
                        if isempty(iter_str), continue; end
                        iter_str = iter_str{end};
                        iter1 = str2double(iter_str);
                        if ~isempty(iter1) && iter1>=0 && iter1>iter
                            iter = iter1;
                        end
                    end
                    %assert( iter>=0, 'no snapshot found' );
                    if iter<0
                        warning( 'No snapshot found. Restored nothing' );
                        return;
                    end
                end
                fn = fn_func(int2str(iter));
            end
            fprintf('Restore from : %s\n', fn);
            assert( exist( fn, 'file' )~=0, 'snaptshot does not exist' );
            obj.solver.restore( fn );
        end
    end
end
