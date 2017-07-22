function b = seqfun( op_func, arg1, varargin )

b = arg1;
for k = 1:length(varargin)
    b1 = op_func(b,varargin{k});
    b = b1;
end

end
