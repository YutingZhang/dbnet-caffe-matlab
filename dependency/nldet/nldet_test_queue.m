function nldet_test_queue( base_dir )

switch_pipeline nldet

if ~exist('base_dir','var') || isempty(base_dir)
    GS = load_global_settings();
    base_dir = GS.NLDET_AUTOTEST_CACHE_DIR;
end

setting_dir = fullfile(base_dir,'setting');
queue_dir   = fullfile(base_dir,'queue');
error_dir   = fullfile(base_dir,'error');
start_dir   = fullfile(base_dir,'start');
done_dir    = fullfile(base_dir,'done');

mkdir_p(setting_dir);
mkdir_p(queue_dir);
mkdir_p(error_dir);
mkdir_p(start_dir);
mkdir_p(done_dir);

    function task_to_error( taskFN, taskFNorig, error_msg )
        if ~exist('error_msg','var') || isempty(error_msg)
            error_msg = '';
        end
        fprintf(2,'nldet_test_queue : error : %s\n', error_msg);
        taskFullFn = fullfile( queue_dir, taskFNorig );
        if exist( taskFullFn, 'file' )
            remove_file(taskFullFn);
            writetextfile( [error_msg sprintf('\n')], ...
                fullfile( error_dir,[ taskFN '.' datestr(now,'yyyy-mm-dd_HH:MM:SS.FFF')] ) );
        end
    end

% CUR_DIR = pwd;

while true
    
    rwt = rand()*20;
    fprintf( 'random wait for %.1f sec\n', rwt );
    pause(rwt);
    
    fprintf( 'Search tasks: %s\n', queue_dir );
    TL = dir( queue_dir );
    TL([TL.isdir]) = [];
    [~,sidx] = sort([TL.datenum]);
    TL = {TL(sidx).name};
    
    task_fn    = [];
    train_id   = [];
    setting_fn = [];
    for k = 1:numel(TL)
        task_fn_orig = TL{k};
        task_fn = task_fn_orig;
        T = regexp( task_fn, '^([0-9]*)-([^\.]*)$', 'once', 'tokens' );
        if isempty(T)
            T = regexp( task_fn, '^([0-9]*)-([^\.]*)\..*$', 'once', 'tokens' );
            if isempty(T)
                task_to_error(task_fn, task_fn_orig, sprintf('Cannot parse filename with regexp: %s\n', task_fn ) ); 
                continue;
            end
            task_fn = [T{1} '-' T{2}];
        end
        train_id = str2double(T{1});
        setting_fn = T{2};
        if ~isempty( train_id ) && ~isempty(setting_fn)
            break;
        end
        task_to_error(task_fn, task_fn_orig, sprintf('Cannot figure out task id from filename: %s', task_fn) ); 
    end
    if isempty( train_id ) || isempty(setting_fn)
        fprintf( 'No task is found. Wait for 30sec' );
        pause(30);
        fprintf('\n');
        continue;
    end
    
    fprintf( 'Process task %d-%s : \n', train_id, setting_fn );
    
    task_full_fn = fullfile(queue_dir,task_fn_orig);
    
    if ~exist( task_full_fn, 'file' )
        fprintf('Task has been done or canceled.\n');
        continue;
    end
    
    setting_path = fullfile(setting_dir,[setting_fn '.m']);
    if ~exist(setting_path,'file')
        task_to_error(task_fn, task_fn_orig, sprintf('Cannot find setting file: %s', setting_path) );
        continue;
    end
    try
        P = refreshed_cached_file_func( @exec2struct, setting_path, 'nldet_test_queue:setting', 20 );
    catch
        task_to_error(task_fn, task_fn_orig, sprintf('Failed to load/parse the setting file.\n %s', setting_path ));
        continue;
    end
    
    start_file_path = fullfile( start_dir, task_fn );
    if exist(start_file_path,'file')
        start_file_info = dir(start_file_path);
        if secdiff( start_file_info.datenum, now ) < 600
            fprintf( 'The job was first started by another worker. Wait for 90sec to avoid initialization conflication. \n' );
            pause(90);
        end
    else
        touch_file(start_file_path);
    end
    
    
    if 1
        try
            [snapshot_iter,test_id] = nldet_test_lastest(train_id, P);
        catch ME
            if ~strcmp( ME.identifier, 'nldetTrainSnapshotCanonicalizeIter:no_snapshot' )
                rethrow(ME);
            end
            task_to_error(task_fn, task_fn_orig, ...
                sprintf('Error occurs during testing. Task filename: %s\nDetails:\n%s', ...
                task_fn, getReport(ME) ));
            remove_file(start_file_path);
            continue;
        end
    else
        [snapshot_iter,test_id] = nldet_test_lastest(train_id, P);
    end
    
    if exist(task_full_fn, 'file')
        D = {train_id, double2str(snapshot_iter), test_id, setting_fn};
        done_fn = sprintf('%d-%s-%d-%s', D{:});
        done_full_fn = fullfile( done_dir,[ done_fn '.' datestr(now,'yyyy-mm-dd_HH:MM:SS.FFF')] );
        writetextfile( cell2csv( D ), done_full_fn );
        remove_file(task_full_fn);
        remove_file(start_file_path);
    end
    
end


end

