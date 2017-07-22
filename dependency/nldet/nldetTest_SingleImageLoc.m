function nldetTest_SingleImageLoc( R, box_limit, loc_dir, do_gtonly, FORCE )

loc_mat_path = fullfile( loc_dir, 'accuracy.mat' );

if ~FORCE && exist(loc_mat_path, 'file')
    fprintf( ' results existed.\n' );
    return;
end

base_loc_param = struct();
base_loc_param.max_proposal_num = box_limit; 
base_loc_param.overlap_threshold = 0.1:0.1:0.7;
base_loc_param.ranking_threshold = 1:10;

ACC = struct();
topOverlap = struct();

loc_param = base_loc_param;

fprintf( '* proposed ------\n' );
loc_param.include_gt       = 0;
loc_param.include_proposed = 1;
[ACC.proposed, topOverlap.proposed] = nldetSingleImageLoc( R, loc_param );

fprintf( '* plusgt   ------\n' );
loc_param.include_gt       = 1;
loc_param.include_proposed = 1;
[ACC.plusgt, topOverlap.plusgt] = nldetSingleImageLoc( R, loc_param );

if do_gtonly
    fprintf( '* gtonly   ------\n' );
    loc_param.include_gt       = 1;
    loc_param.include_proposed = 0;
    [ACC.gtonly, topOverlap.gtonly] = nldetSingleImageLoc( R, loc_param );
end

mkdir_p(loc_dir);

save( loc_mat_path, 'ACC', 'topOverlap', 'base_loc_param' );

T = nldetLocalizationCellTable( ACC, base_loc_param.overlap_threshold, ...
    base_loc_param.ranking_threshold );
writetextfile( cell2csv(T.proposed), fullfile(loc_dir,'proposed.csv') );
writetextfile( cell2csv(T.plusgt),   fullfile(loc_dir,'plusgt.csv') );
meanIoU_str = {
    'proposed', topOverlap.proposed.mean
    'plusgt',   topOverlap.plusgt.mean};
medianIoU_str = {
    'proposed', topOverlap.proposed.median
    'plusgt',   topOverlap.plusgt.median};
if isfield(ACC,'gtonly')
    writetextfile( cell2csv(T.gtonly), fullfile(loc_dir,'gtonly.csv') );
    meanIoU_str = [meanIoU_str
        {'gtonly',   topOverlap.gtonly.mean}];
    medianIoU_str = [medianIoU_str
        {'gtonly',   topOverlap.gtonly.median}];
end
writetextfile( cell2csv(meanIoU_str),   fullfile(loc_dir,'mean_iou.csv') );
writetextfile( cell2csv(medianIoU_str), fullfile(loc_dir,'median_iou.csv') );

