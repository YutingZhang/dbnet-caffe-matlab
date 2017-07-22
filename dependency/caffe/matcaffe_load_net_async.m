function load_func = matcaffe_load_net_async( net_path, batch_size, phase )

if nargin<2
    batch_size = [];
end

if nargin<3
    phase = [];
end

if ischar(phase)
    phase = pbEnum( phase );
end

load_func = @(param) matcaffe_load_net( net_path, batch_size, phase, param );

