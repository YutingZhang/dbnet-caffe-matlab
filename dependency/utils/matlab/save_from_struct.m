function save_from_struct( fn, S, varargin )

save( fn, '-struct', 'S', varargin{:} );

