function varargout = touch_file( varargin )

try
    touch_file_(varargin{:});
    e = 1;
catch
    e = -100;
end

if nargout>0
    varargout{1} = e;
end

end