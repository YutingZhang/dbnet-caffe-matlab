function [Lia, Locb] = ismember_dup( A, B )

[uA,~,uidxA] = unique(B);
uidxA = accumarray( ...
    uidxA, 1:numel(B), [], @(x){x} );
uidxA = cellfun( @sort, uidxA, 'UniformOutput', false );
[isIn, locIn]  = ismember(A,uA);
Locb = cell( size(A) );
Locb(isIn) = uidxA(locIn(isIn));
Lia  = cellfun(@numel, Locb);

