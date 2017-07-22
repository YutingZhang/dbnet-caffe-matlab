function nldet_test_statistics_queue( base_dir )

if ~exist('base_dir','var') || isempty(base_dir)
    GS = load_global_settings();
    base_dir = GS.NLDET_AUTOTEST_CACHE_DIR;
end

switch_pipeline nldet

done_dir = fullfile(base_dir,'done');
loc_dir  = fullfile(base_dir,'stat_loc');
det_dir  = fullfile(base_dir,'stat_det');

mkdir_p(done_dir);
mkdir_p(loc_dir);
mkdir_p(det_dir);

while true
    [candidate_ids, pure_fns] = list_tags_in_dir(done_dir);
    loc_done_ids  = list_tags_in_dir(loc_dir);
    det_done_ids  = list_tags_in_dir(det_dir);
    
    if 0
        both_done_ids = intersect(loc_done_ids,det_done_ids,'stable');
    else
        both_done_ids = loc_done_ids; % current only have loc results
    end
    remain_ids = setdiff(candidate_ids, both_done_ids, 'stable');
    
    if isempty(remain_ids)
        fprintf( 'All done. Wait for 60sec. ' );
        pause(60);
        fprintf( '\n' );
        continue;
    end

    test_id = remain_ids(1);
    pure_fn = pure_fns{ candidate_ids==test_id };
    
    P = struct();
    P.Localization = ~ismember(test_id, loc_done_ids);
    P.Detection    = ~ismember(test_id, det_done_ids);
    
    is_success = 0;
%    try
        nldet_test_statistics(test_id, P);
        is_success = 1;
%    catch
%    end
    
    if is_success
        record_fn = [ pure_fn '.' datestr(now,'yyyy-mm-dd_HH:MM:SS.FFF')];
        touch_file( fullfile(sysStageDir('Test'), ['no-' int2str(test_id)], record_fn) )
        touch_file( fullfile( loc_dir, record_fn) );
        % touch_file( fullfile( det_dir,[ pure_fn '.' datestr(now,'yyyy-mm-dd_HH:MM:SS.FFF')] ) );
    end
end

function [test_ids, pure_fn] = list_tags_in_dir( dir_path )

L = dir(dir_path);
L([L.isdir]) = [];
[~,sidx] = sort([L.datenum]);
L = {L(sidx).name};

test_ids = zeros(numel(L),1);
pure_fn  = cell(numel(L),1);
for k=1:numel(L)
    T = regexp( L{k}, '^([0-9]*)-([0-9\.]*)-([0-9]*)-([^\.]*)\..*$', 'once', 'tokens' );
    if ~isempty(T)
        test_ids(k) = str2double(T{3});
        pure_fn{k}  = sprintf('%s-%s-%s-%s', T{1}, T{2}, T{3}, T{4});
    end
end
pure_fn(test_ids==0) = [];
test_ids(test_ids==0) = [];
