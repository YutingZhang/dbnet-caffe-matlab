function test_ids = nldet_lastest_test_ids(train_ids, test_setting)

switch_pipeline_quiet nldet

test_ids = zeros(size(train_ids));
for k = 1:numel(train_ids)
    ls_patten = fullfile( sysStageDir( 'Test' ), 'no-*', ...
        sprintf( '%d-*-*-*.*', train_ids(k) ) );
    try
        ls_str = ls(ls_patten);
    catch
        % no-such file
        continue;
    end
    FNs = strsplit( ls_str, sprintf('\n') );
    test_ids_k  = zeros(numel(FNs),1);
    test_iter_k = zeros(numel(FNs),1);
    for j = 1:numel(FNs)
        [~,fn_only,~] = fileparts(FNs{j});
        T = regexp( fn_only, ['^[^-]*-([^-]*)-([^-]*)-' test_setting '\..*$'], ...
            'once', 'tokens' );
        if ~isempty(T)
            test_iter_k(j) = str2double(T{1});
            test_ids_k(j)  = str2double(T{2});
        end
    end
    [~,midx] = max(test_iter_k);
    if ~isempty(midx)
        test_ids(k) = test_ids_k(midx);
    end
end