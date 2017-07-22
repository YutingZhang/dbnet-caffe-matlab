function S = loaded2single_var(S)

F=fieldnames(S);
if isscalar(F)
    S = subsref(S,substruct('.',F{1}));
end
