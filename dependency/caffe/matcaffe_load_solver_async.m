function load_func = matcaffe_load_solver_async( solver_path, override_param, ...
    batch_size, output_root )

if nargin<2
    override_param = [];
end

if nargin<3
    batch_size = [];
end

if nargin<4
    output_root = [];
end


load_func = @(param) matcaffe_load_solver( solver_path, override_param, ...
    batch_size, output_root, param );
