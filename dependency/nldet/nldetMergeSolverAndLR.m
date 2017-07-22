function solver_struct1 = nldetMergeSolverAndLR( solver_struct0, lr )

if isempty( solver_struct0 )
    assert( isempty(lr), 'lr should be empty when solver_struct is empty.' );
    solver_struct1 = [];
    return;
end

S = solver_struct0;

if ischar( S.type )
    S.type = {S.type};
end

S.lr_policy = {'fixed'};
S.base_lr = lr;

solver_struct1 = S;
