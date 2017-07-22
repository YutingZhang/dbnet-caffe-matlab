function extra_net_param = nldetNetLoadParamFromBlockDef( block_def )

non_param_field = {
    'def_dir'
    'net_path'
    'batch_size'
    'solver_proto_struct'
    'model_path' 
    };

extra_net_param = partial_struct( block_def, '@exclude', ...
    non_param_field{:} );
