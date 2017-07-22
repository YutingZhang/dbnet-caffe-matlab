function T = lines2string( L )

L = vec(L).';
L(2,:) = {sprintf('\n')};
T = cat(2, L{:});
