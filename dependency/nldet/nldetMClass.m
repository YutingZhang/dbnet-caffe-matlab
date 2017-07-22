function construct_func = nldetMClass( mclass_name )
% construct_func = nldetMClass( mclass_name )

a = mclass_name;
a = nldetMClassAlias(a);
construct_func = eval( sprintf( '@(varargin) %s(varargin{:})', a ) );

