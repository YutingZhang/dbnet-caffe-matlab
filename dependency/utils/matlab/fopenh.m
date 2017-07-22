function fh = fopenh( varargin )

fid = fopen(varargin{:});
if fid>0
    fh = struct('id',fid, 'cleanup', onCleanup(@() fclose(fid)) );
else
    fh = struct('id',[], 'cleanup', [] );
end
