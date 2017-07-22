function R = nldet_best_model_for_localization( disabled_train_ids )

if nargin<1
    disabled_train_ids = [];
end

switch_pipeline nldet

BASE_DIR = sysStageDir('Test');

L = dir(fullfile(BASE_DIR,'no-*'));
L(~[L.isdir]) = [];
L = {L.name};

R = struct();
R.proposed = [];
R.plusgt   = [];
R.gtonly   = [];
for k = 1:numel(L)
    T = regexp( L{k}, '^no-([0-9]*)$', 'once', 'tokens' );
    if isempty(T), continue; end
    
    test_id = str2double( T{1} );
    
    train_test_tag = dir( fullfile( BASE_DIR, L{k}, ['*-*-' T{1} '-*.*'] ) );
    if isempty(train_test_tag)
        train_id = 0;
        train_test_tag = '';
    else
        if ~isscalar(train_test_tag)
            [~,sidx] = sort( [train_test_tag.datenum], 'descend' );
            train_test_tag = train_test_tag(sidx(1));
        end
        % assert( isscalar(train_test_tag), 'found multiple train_test_tag' );
        train_test_tag = train_test_tag.name;
        train_id = regexp( train_test_tag, '([0-9]*)-[0-9\.]*-[0-9]*-.*', 'once', 'tokens' );
        train_id = str2double(train_id{1});
        if ismember(train_id, disabled_train_ids)
            continue;
        end
    end
    
    acc_base_dir = fullfile( BASE_DIR, L{k}, 'localization' );
    if ~exist(acc_base_dir, 'dir'), continue; end
    
    D = dir(fullfile(acc_base_dir,'boxes_*'));
    D(~[D.isdir]) = [];
    D = {D.name};
    
    for j = 1:numel(D)
    
        T = regexp( D{j}, '^boxes_(.*)$', 'once', 'tokens' );
        if isempty(T), continue; end
        num_boxes = str2double( T{1} );
        if isnan(num_boxes), continue; end
        
        acc_path = fullfile( acc_base_dir, D{j}, 'accuracy.mat' );
        
        ACC = cached_file_func( @load, acc_path, 'nldet-accuracy-results', 1000 );
        ACC = ACC.ACC;
        
        if size(ACC.proposed,2)~=10, continue; end % ignore old results in different format
        
        if isempty( R.proposed )
            R.proposed.acc = ACC.proposed;
            R.proposed.test_id   = repmat( test_id, size(R.proposed.acc) );
            R.proposed.num_boxes = repmat( num_boxes, size(R.proposed.acc) );
            R.proposed.train_test_tag = repmat( {train_test_tag}, size(R.proposed.acc) );
        else
            refresh_idxb = ( ACC.proposed > R.proposed.acc );
            R.proposed.acc(refresh_idxb(:)) = ACC.proposed(refresh_idxb(:));
            R.proposed.test_id(refresh_idxb(:))   = test_id;
            R.proposed.num_boxes(refresh_idxb(:)) = num_boxes;
            R.proposed.train_test_tag(refresh_idxb(:)) = {train_test_tag};
        end

        if isempty( R.plusgt )
            R.plusgt.acc = ACC.plusgt;
            R.plusgt.test_id   = repmat( test_id, size(R.plusgt.acc) );
            R.plusgt.num_boxes = repmat( num_boxes, size(R.plusgt.acc) );
            R.plusgt.train_test_tag = repmat( {train_test_tag}, size(R.plusgt.acc) );
        else
            refresh_idxb = ( ACC.plusgt > R.plusgt.acc );
            R.plusgt.acc(refresh_idxb(:)) = ACC.plusgt(refresh_idxb(:));
            R.plusgt.test_id(refresh_idxb(:))   = test_id;
            R.plusgt.num_boxes(refresh_idxb(:)) = num_boxes;
            R.plusgt.train_test_tag(refresh_idxb(:)) = {train_test_tag};
        end

        if ~isfield( ACC, 'gtonly' ), continue; end

        if isempty( R.gtonly )
            R.gtonly.acc = ACC.gtonly;
            R.gtonly.test_id   = repmat( test_id, size(R.gtonly.acc) );
            R.gtonly.num_boxes = repmat( num_boxes, size(R.gtonly.acc) );
            R.gtonly.train_test_tag = repmat( {train_test_tag}, size(R.gtonly.acc) );
        else
            refresh_idxb = ( ACC.gtonly > R.gtonly.acc );
            R.gtonly.acc(refresh_idxb(:)) = ACC.gtonly(refresh_idxb(:));
            R.gtonly.test_id(refresh_idxb(:))   = test_id;
            R.gtonly.num_boxes(refresh_idxb(:)) = num_boxes;
            R.gtonly.train_test_tag(refresh_idxb(:)) = {train_test_tag};
        end
        
    end

end
