function varargout = nldet_print_test_results( ...
    test_ids, num_boxes, iou_th, test_titles )
% 

if ~exist('iou_th','var') || isempty(iou_th)
    iou_th = 0.1:0.1:0.7;
end

switch_pipeline_quiet nldet

numTest = numel(test_ids);

T_col = [ { 'Test ID' } , ...
    arrayfun( @(a) sprintf('IoU@%g', a), iou_th, 'UniformOutput', 0 ), ...
    { 'median IoU', 'mean IoU' } ];
if exist('test_titles', 'var') && ~isempty(test_titles)
    assert(numel(test_titles)==numel(test_ids), ...
        'lengths of test_titles and test_ids should be the same');
    T_row = vec(test_titles);    
else
    T_row = arrayfun( @(a) sprintf( 'no-%d', a ), vec(test_ids), ...
        'UniformOutput', 0 );    
end
C     = cell( numTest, numel(iou_th)+2 );

acc_fn = sprintf( 'localization/boxes_%d/accuracy.mat', num_boxes );

for k = 1:numTest
    acc_path = fullfile( sysStageDir( 'Test' ), sprintf('no-%d',test_ids(k)), acc_fn );
    if ~exist( acc_path, 'file' ),
        warning( 'Cannot find accuracy mat for %s', T_row{k} );
        continue;
    end
    R = load(acc_path);
    [has_iou, iou_pos] = ismembertol( iou_th, R.base_loc_param.overlap_threshold );
    assert( all(has_iou), 'Not all iou_th exist' );
    C_k_acc = R.ACC.proposed(iou_pos,1).'*100;
    C_k_ov  = [ R.topOverlap.proposed.median, R.topOverlap.proposed.mean ];
    C(k,:) = [arrayfun(@(a) sprintf('%.1f',a), C_k_acc, 'UniformOutput', 0 ), ...
        arrayfun(@(a) sprintf('%.3f',a), C_k_ov, 'UniformOutput', 0 ) ];
end

T = [T_col; [T_row,C]];

T_str = cell2csv(T);

if nargout==0
    fprintf('%s\n', T_str);
else
    varargout = {T_str};
end
