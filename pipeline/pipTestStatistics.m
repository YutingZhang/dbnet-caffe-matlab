% Pipline stage: TestStatistics

%% Parameters
% define parameters that do not affect output here
% e.g. OTHER_DEFAULT_PARAM.A = 0
OTHER_DEFAULT_PARAM = struct();

OTHER_DEFAULT_PARAM.Localization = 1;
OTHER_DEFAULT_PARAM.Detection    = 1;
OTHER_DEFAULT_PARAM.TextSimilarityThreshold = 1; % [1 0.15];

OTHER_DEFAULT_PARAM.ProposedBoxes_Limit = 500; %[Inf, 500, 200, 100];

OTHER_DEFAULT_PARAM.FORCE = 0;

PARAM = xmerge_struct( OTHER_DEFAULT_PARAM, PARAM );



%% initialization for stage


%% Start your code here

% check if test results exist
result_path = fullfile( STAGE_SPECIFIC_DIR, 'results.v7.mat' );

result_file_existed = boolean(exist(result_path,'file'));
numTotalRetry = 5;
for k = 1:numTotalRetry
    if result_file_existed, break; end
    fprintf( 'results.v7.mat not found. Retry. %d / %d \n', k, numTotalRetry );
    pause(60);
    result_file_existed = boolean(exist(result_path,'file'));
end
assert( result_file_existed, 'results.v7.mat does not exist' );

% initialize gt loaders
data_loader = nldetGetDataLoaderConstructor( SPECIFIC_DIRS.PrepDataset );
gt_loader   = data_loader.gt_loader( PARAM.Test_DataSubset, ...
    SPECIFIC_DIRS.PrepDataset );
text_comp_func = [];
if ismethod(gt_loader,'text_comp');
    text_comp_func = @(varargin) gt_loader.text_comp(varargin{:});
end

% load test results
t1 = tic_print( 'Load test results : ' );

result_file_info = dir(result_path);
rfi_datenum = result_file_info.datenum;
rfi_datenum0 = 0;
while rfi_datenum~=rfi_datenum0 && secdiff( rfi_datenum, now ) < 120
    pause(120);
    result_file_info = dir(result_path);
    rfi_datenum0 = rfi_datenum;
    rfi_datenum  = result_file_info.datenum;
end

R = load7_single( result_path );
toc_print(t1);

% load test
if PARAM.Localization
    
    fprintf( '========================================\n' );
    fprintf( '============== Localization ============\n' );
    fprintf( '========================================\n' );
    
    for j = 1:numel( PARAM.TextSimilarityThreshold )
        
        sim_th = PARAM.TextSimilarityThreshold(j);
        
        if sim_th == 1
            R1 = nldetAddDontCare2Result( R );
            loc_base_dir = fullfile( STAGE_SPECIFIC_DIR, sprintf('localization' ) );
        else
            if isempty(text_comp_func)
                warning( 'no text similarity score exists, ignored' );
                continue;
            end
            R1 = nldetAddDontCare2Result( R, text_comp_func, sim_th );
            loc_base_dir = fullfile( STAGE_SPECIFIC_DIR, sprintf('localization-sim_%g', sim_th) );
        end
     
        PBL = sort( unique( PARAM.ProposedBoxes_Limit ), 'descend' );
        for k = 1:numel(PBL);

            box_limit = PBL(k);
            fprintf( 'sim_th %g ; boxes %d ================\n', sim_th, box_limit );

            loc_dir = fullfile( loc_base_dir, sprintf('boxes_%d', box_limit ) );
            nldetTest_SingleImageLoc( R1, box_limit, loc_dir, k==1, PARAM.FORCE );

        end

    end
end


