function writetextfile( T, fn, is_append )

if nargin<3
    is_append = 0;
end

if iscell(T)
    T = lines2string(T);
end

verbose = 0;
if ischar(is_append)
    switch is_append
        case 'w-' % no overwrite, no verbose
            if exist( fn, 'file' ), return; end
            fopen_tag = 'w';
        case 'w-v' % no overwrite, but verbose
            if exist( fn, 'file' ),
                fprintf( 'file existed, did not overwrite: %s\n', fn );
                return;
            end
            fopen_tag = 'w';
            verbose   = 1;
        case 'a'
            fopen_tag = 'a';
        case 'w'
            fopen_tag = 'w';
        case 'wv'
            fopen_tag = 'w';
            verbose   = 1;
        otherwise
            error( 'Invalid open tag' );
    end
elseif is_append
    fopen_tag = 'a';
else
    fopen_tag = 'w';
end

fid = fopen( fn, fopen_tag );
if fid>0
    closefile_obj = onCleanup( @() fclose(fid) );
end

fprintf( fid, '%s', T );

if verbose
    fprintf( 'file written: %s\n', fn );
end

