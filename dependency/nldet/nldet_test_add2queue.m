function nldet_test_add2queue( test_id, test_setting )

GS = load_global_settings();
base_dir = GS.NLDET_AUTOTEST_CACHE_DIR;

setting_dir = fullfile(base_dir,'setting');

assert( boolean(exist(fullfile(setting_dir,[test_setting '.m']),'file')), ...
    'no such test_setting' );

queue_dir = fullfile(base_dir,'queue');
mkdir_p(queue_dir);

task_fn = sprintf( '%d-%s', test_id, test_setting );
touch_file( fullfile(queue_dir, task_fn) );
