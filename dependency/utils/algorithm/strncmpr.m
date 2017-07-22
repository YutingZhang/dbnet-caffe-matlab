function c = strncmpr( a, b, n )
% reverse strncmp

if ischar(a)
    a = fliplr(a);
elseif iscell(a)
    a = cellfun( @fliplr, a, 'UniformOutput', 0 );
else
    error('Invalid input');
end

if ischar(b)
    b = fliplr(b);
elseif iscell(b)
    b = cellfun( @fliplr, b, 'UniformOutput', 0 );
else
    error('Invalid input');
end

c = strncmp(a,b,n);
