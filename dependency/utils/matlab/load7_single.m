function a = load7_single(fn, opt_var_name)

if nargin<2
    S = load7(fn);
else
    S = load7(opt_var_name);
end

F = fieldnames(S);

assert(isscalar(F), 'must only load a single variable');

a = subsref(S, substruct('.',F{1}));
