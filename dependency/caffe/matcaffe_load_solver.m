function solver = matcaffe_load_solver( solver_path, override_param, ...
    batch_size, output_root, extra_net_param )
% solver = matcaffe_load_solver( solver_path, override_param, ...
%     [train_batch_size, test_batch_size1, test_batch_size2], output_root, extra_net_params )
%  batch_size: (1[train]+num_of_test_net)*(1 or num_input_blob)

if ~exist('override_param','var') || isempty(override_param)
    override_param = struct();
end

assert( all(batch_size>0) && all(uint64(batch_size(:))==batch_size(:)), 'batch_size must be positive integer' );
numTestNets = size(batch_size,1) - 1;

if ~exist('extra_net_param','var') || isempty(extra_net_param)
    extra_net_param = struct();
end

[tmp_dir, tmp_dir_cleanup] = get_tmp_folder( 'caffe_load_solver' );

if ischar( solver_path )
    assert( exist( solver_path, 'file' )~=0, 'cannot find solver file' );
    S = prototxt2struct( solver_path );
else
    assert(isscalar(solver_path) && isstruct(solver_path), ...
        'invalid solver specification' );
    S = solver_path;
end
S = xmerge_struct( 'always', 'always', S, override_param );

snapshot_interval = default_eval( 'S.snapshot', 0 );
if ~exist('output_root','var') || (~ischar(output_root) && isempty(output_root))
    if ischar( solver_path )
        output_root = fileparts(solver_path);
    else
        if snapshot_interval>0
            error( 'must specify output_root when directly using solver proto struct' );
        end
    end
end


test_iter = default_eval('S.test_iter',[]);
if numTestNets>0
    if isscalar(test_iter)
        test_iter = repmat( test_iter, 1, numTestNets );
    end
    assert( numel(test_iter)==numTestNets, 'num of test_iter does not match with number of test nets' );
end

if snapshot_interval>0
    snapshot_prefix = default_eval( 'S.snapshot_prefix{1}', 'model' );
    snapshot_prefix = fullfile(output_root,snapshot_prefix);
    S.snapshot_prefix = {snapshot_prefix};
else
    S.snapshot_prefix = [];
end
S.test_iter = []; % always disable testing during training
S.test_interval = [];

net_path = S.net{1};
assert( ischar(net_path) && ~isempty(net_path), 'net_path should be non-empty string' );
if (net_path~='/')
	net_path = fullfile( fileparts( solver_path ), net_path );
end

assert( exist( net_path, 'file' )~=0, 'cannot find net file' );

revised_net_paths = cell(1,numTestNets+1);
[revised_net_paths{:}] = matcaffe_load_net( net_path, batch_size, tmp_dir, extra_net_param );

S.net = revised_net_paths(1);
revised_solver_path = fullfile(tmp_dir,'solver.prototxt');
struct2prototxt( S, revised_solver_path );

caffe_solver = caffe.Solver(revised_solver_path);
solver = matcaffeSolver(caffe_solver, true);

for k = 1:numTestNets
    caffe_net = caffe.Net( revised_net_paths{1+k}, 'test' );
    net = matcaffeNet( caffe_net, true );
    solver.add_test_net(net, test_iter(k));
end

