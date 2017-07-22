function [X, varargout] = caffeproto_replace_func_rm_if( varargin )

if ischar(varargin{1})
    if strcmp( varargin{1}, 'adjacent' )
        X = [0];
        varargout = {};
    elseif strcmp( varargin{1}, 'extend' )
        X = {};
    else
        error( 'Unrecognized mode: %s', varargin{1} );
    end
    return;
end

subS = varargin{1};
match_func = varargin{end}{1};
ismatched = try_to_eval( 'match_func( subS(1) )', false );
if ismatched
    X = struct([]);
else
    X = [];
end
