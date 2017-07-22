function save7_single( fn, a )

if ischar(a)
    n = a;
    a = evalin('caller',a);
else
    n = inputname(2);
    if isempty(n), n = 'a'; end
end

assignhere( n, a );

save7(fn, n);
