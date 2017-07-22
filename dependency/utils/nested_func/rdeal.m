function varargout = rdeal( varargin )

if nargin == 0
    if nargout>0
        varargout = cell(1,nargout);
    end
    return;
end

for k=1:ceil(nargout/nargin)
    st = (k-1)*nargin;
    en = min(k*nargin,nargout);
    varargout(st+1:en) = varargin(1:en-st);
end


