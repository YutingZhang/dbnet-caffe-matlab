function nldet_test_statistics_add2queue( test_id )

GS = load_global_settings();
base_dir = GS.NLDET_AUTOTEST_CACHE_DIR;

done_dir = fullfile(base_dir,'done');
mkdir_p(done_dir);

task_fn = sprintf( '0-0-%d-external.%s', test_id, ...
    datestr(now,'yyyy-mm-dd_HH:MM:SS.FFF') );
touch_file( fullfile(done_dir, task_fn) );
