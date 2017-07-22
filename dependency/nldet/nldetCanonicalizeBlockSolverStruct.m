function S1 = nldetCanonicalizeBlockSolverStruct( S0, Sdef, HasCaffe )

if isempty(HasCaffe)
    S1 = [];
else
    S1 = matcaffeCanonicalizeSolverStruct( S0, Sdef );
    S1.type = S1.type{1};
end
