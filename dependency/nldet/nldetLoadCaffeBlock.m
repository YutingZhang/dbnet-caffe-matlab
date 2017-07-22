function block = nldetLoadCaffeBlock( block_def )
% block = nldetLoadCaffeBlock( block_def )
%     block_def.def_dir
%     block_def.batch_size
%     block_def.outputs
%     block_def.solver_proto_struct % empty for test mode
%     block_def.model_path
%     block_def.place_holders

is_solver = ~isempty( block_def.solver_proto_struct );
if is_solver
    assert( isscalar(block_def.solver_proto_struct) && isstruct(block_def.solver_proto_struct), ...
        'invalid solver_proto_struct' );
    % use solver to load
    S = block_def.solver_proto_struct;
    net_path = absolute_path(block_def.net_path);
    S.snapshot  = 0;
    S.iter_size = 1;
    S.net = {net_path};
    mn = matcaffe_load_solver_async(S,[],block_def.batch_size);
else
    if block_def.is_train
        net_phase_str = 'train';
    else
        net_phase_str = 'test';
    end
    mn = matcaffe_load_net_async(block_def.net_path,vec(block_def.batch_size), ...
        pbEnum(net_phase_str));
end

extra_net_param = nldetNetLoadParamFromBlockDef( block_def );

block = matcaffeBlock(mn, extra_net_param);

if ~isempty(block_def.model_path)
    block.net.copy_from( block_def.model_path );
end
