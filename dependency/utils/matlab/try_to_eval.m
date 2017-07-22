function varargout = try_to_eval( expStr, varargin )

defVal = varargin;
defVal = [defVal, cell(1,nargout-length(varargin)) ];

varargout = cell(1,nargout);

try
    [varargout{:}] = evalin( 'caller', expStr );
catch 
    varargout = defVal;
end
