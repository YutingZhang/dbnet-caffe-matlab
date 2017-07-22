function [X, varargout] = caffeproto_replace_func_rm( varargin )

if ischar(varargin{1})
    if strcmp( varargin{1}, 'adjacent' )
        X = [0 1; 0 0];
        varargout = {1,2};
    elseif strcmp( varargin{1}, 'extend' )
        X = {};
    else
        error( 'Unrecognized mode: %s', varargin{1} );
    end
    return;
end

subS      = varargin{1};
subSS     = struct( 'layer', subS(1) );
ismatched = caffeproto_subnet(subSS,varargin{end}{:});
ismatched = ismatched && length(subS(1).bottom)==length(subS(2).bottom);
if ismatched
    X = subS(2);
    X.bottom = subS(1).bottom;
else
    X = [];
end
