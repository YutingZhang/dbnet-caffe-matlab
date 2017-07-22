function a = load7_try_single(fn, opt_var_name)

if nargin<2
    S = load7(fn);
else
    S = load7(opt_var_name);
end

F = fieldnames(S);

if isscalar(F) 
    a = subsref(S, substruct('.',F{1}));
else
    a = S;
end
