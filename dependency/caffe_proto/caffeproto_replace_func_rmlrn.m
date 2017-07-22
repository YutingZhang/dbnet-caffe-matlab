function [X, varargout] = caffeproto_replace_func_rmlrn( varargin )

if strcmp( varargin{1}, 'extend' )
	X = { {'rm',@(a) strcmp(a.type,'LRN')} };
else
	error( 'Unrecognized mode: %s', varargin{1} );
end
