function [net, varargout] = nldetLoadBlock( ...
    def_dir, batch_size, solver_proto_struct, PARAM, EXTRA_NET_PARAM_or_FUNC )
% net = nldetLoadBlock( block_type, def_dir, max_batch_size, ...
%          solver_proto_struct, PARAM, EXTRA_NET_PARAM )
% [net,trainer] = ___
% if empty solver_proto_struct is specified, the network is load in test mode

varargout = {};

if ~exist('PARAM', 'var') || isempty(PARAM)
    PARAM = struct();
end

if ~exist('EXTRA_NET_PARAM_or_FUNC', 'var') || isempty(EXTRA_NET_PARAM_or_FUNC)
    EXTRA_NET_PARAM_or_FUNC = struct();
end

is_train_proto = ~isempty( solver_proto_struct );
is_solver = is_train_proto;
if is_train_proto
    if ischar( solver_proto_struct )
        if strcmp( solver_proto_struct, 'noupdate' )
            is_solver = 0;
        else
            error( 'Unrecognized solver_proto_struct string' );
        end
        solver_proto_struct = []; % clear after setting the flags
    else
        assert( isscalar(solver_proto_struct) && isstruct(solver_proto_struct), ...
            'solver_proto_struct should be either a scalar struct or ''noupdate''' );
    end
end
if is_solver
    proto_phase_str = 'train';
    assert( isscalar(solver_proto_struct) && isstruct(solver_proto_struct), ...
        'invalid solver_proto_struct' );
    solver_proto_struct.error_on_nan_loss = false;
    solver_proto_struct.error_on_inf_loss = false;
else
    if is_train_proto
        proto_phase_str = 'train';
    else
        proto_phase_str = 'test';
    end
end
BP = nldetBlockInitParamFromDir( def_dir, proto_phase_str );

block_def = struct();
block_def.def_dir       = def_dir;
block_def.net_path      = BP.net_path;
block_def.batch_size    = batch_size;
block_def.solver_proto_struct = solver_proto_struct;
block_def.model_path    = BP.model_path;
block_def.outputs       = BP.outputs;
block_def.place_holders = PARAM;
block_def.is_train      = is_train_proto;

if ~isempty(BP.mclass)
    block_def2 = default_eval( sprintf('%s.extra_net_param', BP.mclass), struct() );
    block_def = xmerge_struct( block_def, block_def2 );
end

if isstruct(EXTRA_NET_PARAM_or_FUNC)
    block_def1 = EXTRA_NET_PARAM_or_FUNC;
elseif isa( EXTRA_NET_PARAM_or_FUNC, 'function_handle' )
    block_def1 = EXTRA_NET_PARAM_or_FUNC(block_def);
else
    error( 'Invalid EXTRA_NET_PARAM_or_FUNC' );
end
block_def = xmerge_struct( block_def, block_def1 );

preload_caffe_net = ~BP.seal_caffe_net && ~isempty(BP.net_path);
if preload_caffe_net
    block = nldetLoadCaffeBlock( block_def );
    block.aux.block_def = block_def;
end

if isempty(BP.mclass)
    net = block;
else
    cls_construct_func = nldetMClass(BP.mclass);
    if preload_caffe_net
        net = cls_construct_func( block, PARAM );
    else
        net = cls_construct_func( block_def, PARAM );
    end
end

if nargout>=2
    assert( ~isempty(solver_proto_struct), 'Trainer can only be generated in train model' );
    if isempty( BP.trainer_mclass )
        varargout{1} = [];
    else
        trainer_construct_func = nldetMClass(BP.trainer_mclass);
        varargout{1} = trainer_construct_func(net,block_def);
    end
end