function varargout = cached_file_func( fileFunc, filePath, ...
    cacheName, maxCacheSize, varargin )

if ~exist( 'cacheName', 'var' ) || isempty(cacheName)
    cacheName = '[default]';
end

if ~exist( 'maxCacheSize', 'var' ) || isempty(maxCacheSize) || maxCacheSize<=0
    maxCacheSize = [];
end

persistent cached_file_map
global cached_file_do_func_flag

cached_file_do_func_flag = false;

if isempty(cached_file_map)
    cached_file_map = containers.Map('KeyType','char','ValueType','any');
end

if isKey( cached_file_map, cacheName )
    s = cached_file_map( cacheName );
    if isempty(maxCacheSize), maxCacheSize = s.maxCacheSize;
    elseif maxCacheSize ~= s.maxCacheSize;
        s.maxCacheSize = maxCacheSize;
        cached_file_map( cacheName ) = s;
    end
    m = s.cachedMap;
else
    if isempty(maxCacheSize), maxCacheSize = 100; end
    if maxCacheSize<=0, return; end
    m = containers.Map('KeyType','char','ValueType','any');
    cached_file_map( cacheName ) = struct( ...
        'cachedMap', m, ...
        'maxCacheSize', maxCacheSize );
end

varargout = cell(1,nargout);

if ~isempty( filePath )
    filePath = absolute_path( filePath );
    % different number of output may have different behaviour
    fileKey = [ int2str(nargout) ':' filePath ];
    if isempty(varargin)
        % logically this is for file loading
        assert( exist(filePath,'file')~=0, 'file does not exist' );
        
        a = dir(filePath);
        need_update = 1;
        if m.isKey( fileKey )
            f = m(fileKey);
            if f.datenum == a.datenum
                need_update = 0;
                varargout = f.content;
            end
        end

        if need_update
            f = struct( 'datenum', a.datenum, 'content', { [] } );
            [varargout{:}] = fileFunc( filePath );
            f.content  = varargout;
            m(fileKey) = f;
        end
        cached_file_do_func_flag = need_update;
    else
        % this is for file saving (for content update)
        [varargout{:}] = fileFunc( filePath, varargin{:} );
        a = dir(filePath);
        f = struct( 'datenum', a.datenum, 'content', { varargin } );
        m(fileKey) = f;    
        cached_file_do_func_flag = true;
    end

end

if m.Count > maxCacheSize
    if maxCacheSize<=0
        remove( cached_file_map, cacheName );
    else
        K = keys(m);
        V = values(m);
        V = cellfun( @(a) a.datenum, V );
        [~,r] = sort(V,'descend');
        remove(m,K(r(maxCacheSize+1:end)));
    end
end


