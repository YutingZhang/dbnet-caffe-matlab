function varargout = nldet_cleanup_checkpoints

assert( strcmp( cur_pipeline_name, 'nldet' ), 'nldet pipeline is not activated' );

train_test_tags = get_existing_train_test_tag;

% find tested checkpoints to preserve
tested_train_id_iter = zeros(numel(train_test_tags),2);
for k = 1:numel(train_test_tags)
    T = regexp( train_test_tags{k}, '^no-[0-9]*/([0-9]*)-([0-9\.]*)-[0-9]*-.*\..*$', 'once', 'tokens' );
    if ~isempty(T)
        tested_train_id_iter(k,1) = str2double(T{1});
        tested_train_id_iter(k,2) = str2double(T{2});
    end
end
tested_train_id_iter( tested_train_id_iter(:,1)==0,: ) = [];

% find all existing checkpoints
existing_train_id_iters = get_existing_train_id_iters;
existing_tested_train_id_iter = zeros( numel(existing_train_id_iters), 2 );

for k = 1:numel(existing_train_id_iters)
    T = regexp( existing_train_id_iters{k}, '^no-([0-9]*)/iter_([0-9]*)$', 'once', 'tokens' );
    if ~isempty(T)
        existing_tested_train_id_iter(k,1) = str2double(T{1});
        existing_tested_train_id_iter(k,2) = str2double(T{2});
    end
end
existing_tested_train_id_iter( existing_tested_train_id_iter(:,1)==0, : ) = [];

% find latest checkpoints to preserve
unique_train_id = unique( existing_tested_train_id_iter(:,1) );
max_idxb = false( size(existing_tested_train_id_iter,1), 1 );
for k = 1:numel(unique_train_id)
    cur_idxes  = find( existing_tested_train_id_iter(:,1)==unique_train_id(k) );
    cur_iters = existing_tested_train_id_iter( cur_idxes, 2 );
    [~,sidx] = max(cur_iters);
    max_idxb( cur_idxes(sidx) ) = true;
end
redundant_tested_train_id_iter = existing_tested_train_id_iter(~max_idxb,:);

redundant_tested_train_id_iter = setdiff(redundant_tested_train_id_iter, ...
    tested_train_id_iter, 'rows');

if nargout > 0
    varargout = {redundant_tested_train_id_iter};
else
    fprintf( 'Are you sure to remove %d checkpoints ? (yes to continue, otherwise abort) ', ...
        size(redundant_tested_train_id_iter,1) );
    c = input('','s');
    if strcmpi(c,'yes')
        for k = 1:size(redundant_tested_train_id_iter,1)
            train_id   = redundant_tested_train_id_iter(k,1);
            train_iter = redundant_tested_train_id_iter(k,2);
            fn = fullfile( sysStageDir('Train'), sprintf('no-%d/iter_%d', ...
                train_id, train_iter) );
            fprintf( '%s : ', fn );
            remove_file_recursively( fn );
            fprintf( 'removed\n' );
        end
    end
end


function train_id_iters = get_existing_train_id_iters

CUR_DIR = pwd;
back2pwd_cleanup = onCleanup( @() cd(CUR_DIR) );
cd( sysStageDir('Train') );

train_id_iter_ls = ls( '-d', 'no-*/iter_*' );

train_id_iters = strsplit(train_id_iter_ls, sprintf('\n'));
train_id_iters = cellfun( @strtrim, train_id_iters, 'UniformOutput', 0 );
train_id_iters( cellfun(@isempty, train_id_iters) ) = [];


function train_test_tags = get_existing_train_test_tag

CUR_DIR = pwd;
back2pwd_cleanup = onCleanup( @() cd(CUR_DIR) );

cd( sysStageDir('Test') );

tested_ls = ls( '*/*-*-*-*.*' );

train_test_tags = strsplit(tested_ls, sprintf('\n'));
train_test_tags = cellfun( @strtrim, train_test_tags, 'UniformOutput', 0 );
train_test_tags( cellfun(@isempty, train_test_tags) ) = [];
