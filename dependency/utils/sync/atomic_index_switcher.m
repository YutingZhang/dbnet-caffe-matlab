function varargout = atomic_index_switcher( index_file_path, cmd, task_total, task_time_out, task_id )

assert( ischar(index_file_path), 'index_file_path should be a string' );
assert( ischar(cmd), 'cmd should be a string' );
assert( isnumeric(task_total), 'task_total should be numeric' );
assert( isnumeric(task_time_out), 'task_time_out should be numeric' );
assert( isnumeric(task_id), 'tast_id should be numeric' );

web_disable_interval = 5*60;
time_out = 5;

persistent web_disable_at
if ~isempty(web_disable_at) && toc(web_disable_at)>web_disable_interval 
    web_disable_at = [];
end

GS = load_global_settings;

is_success = 0;
if isempty(web_disable_at) && ...
        isfield(GS,'AUTOMIC_INDEX_WEB_URL') && ...
        ~isempty(GS.AUTOMIC_INDEX_WEB_URL)
    
    POST_CONTENT = {'index_file_path', index_file_path, ...
        'cmd',cmd, ...
        'task_total',int2str(task_total), ...
        'task_time_out',int2str(task_time_out),...
        'tast_id',int2str(task_id)};
    [R, status] = urlread( GS.AUTOMIC_INDEX_WEB_URL, 'Post', POST_CONTENT, 'Timeout', time_out );

    if status
        R = strsplit( R, sprintf('\n') );
        if length(R)>=7
            try
                R = cellfun(@str2double,R);
                R = num2cell(R);
                is_success = 1;
            catch
            end
        end
    end
    
    if ~is_success
        warning( 'atomic_index_switcher: web service unavailabe use local fallback' );
        web_disable_at = tic;
    end
    
end

if ~is_success
    R = cell(1,7);
    [R{:}] = atomic_index_( index_file_path, cmd, task_total, task_time_out, task_id );
end

varargout = reshape( R, 1, numel(R) );

end
