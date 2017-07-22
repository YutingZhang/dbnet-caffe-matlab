function [ACC, topOverlap] = nldetSingleImageLoc( formatted_results, PARAM )

if ~exist('PARAM','var') || isempty( PARAM )
    PARAM = struct();
end

Pdef.overlap_threshold = 0.1:0.1:0.7;
Pdef.ranking_threshold = 1:10;
Pdef.include_gt       = 0;
Pdef.include_proposed = 1;
Pdef.max_proposal_num = inf;
Pdef.result_source    = 'scores'; % 'scores', 'oracle', 'random', 'average'

PARAM = xmerge_struct(Pdef,PARAM);

assert( ismember(PARAM.result_source, ...
    {'scores', 'oracle', 'random', 'average'}), ...
    'unrecognized result source' );

rs = PARAM.result_source;
if strcmp(rs,'average')
    PARAM.ranking_threshold = 1;
end

tic_toc_print;

numImages = length(formatted_results);
ACC = cell(numImages,1);
topOV = cell(numImages,1);
numPhrases = zeros(numImages,1);
parfor k = 1:numImages
% for k = 1:numImages
    tic_toc_print( '%d / %d\n', k, numImages );
    [ACC_k, topOV_k, numPhrases_k]  = nldetSingleImageLoc_Single( formatted_results(k), PARAM );
    ACC{k}   = ACC_k*numPhrases_k;
    numPhrases(k) = numPhrases_k;
    topOV{k} = topOV_k;
end
ACC = seqfun( @plus, ACC{:} )/sum(numPhrases);
topOV = cat( 2, topOV{:} );
topOverlap = struct();
topOverlap.data = topOV;
topOverlap.median = median(topOV);
topOverlap.mean   = mean(topOV);

function [ACC, topOverlap, numPhrases] = nldetSingleImageLoc_Single( R, PARAM )

iouT  = PARAM.overlap_threshold;
rankT = PARAM.ranking_threshold;
includeGT = PARAM.include_gt;
includeProposed = PARAM.include_proposed;
numMaxProposals = PARAM.max_proposal_num;

R.is_gt = boolean(R.is_gt);

% exclude fully negative phrase
phrase_has_gt = any(R.is_gt,1);
R.is_gt(:,~phrase_has_gt) = [];
R.labels(:,~phrase_has_gt) = [];
R.scores(:,~phrase_has_gt) = [];


if ~isfield(R,'dontcare')
    R.dontcare = false(size(R.is_gt));
end
gt_idxb = any(R.is_gt,2);
dontcare_idxb = any( R.dontcare, 2 );
proposed_idxb = ~( gt_idxb | dontcare_idxb );

% gt_idx = find( gt_idxb );
% gt_boxes = R.boxes(gt_idxb,:);
% ov = zeros(size(gt_boxes,1), size(R.boxes,1));
% for i = 1:size(gt_boxes,1)
%     ov(i,:) = PascalOverlap( gt_boxes(i,:), R.boxes);
% end

numPhrases = size(R.is_gt,2);
if strcmp( PARAM.result_source, 'average' )
    ACC = zeros(length(iouT),length(rankT),numPhrases);
else
    ACC = false(length(iouT),length(rankT),numPhrases);
end
topOverlap = zeros( 1, numPhrases ); 
for j = 1:numPhrases
    valid_idxb = ~isnan(R.scores(:,j)); % remove untested scores
    if ~includeGT
        valid_idxb = valid_idxb & ~( gt_idxb | dontcare_idxb );
    else
        valid_idxb = valid_idxb & ~( ~gt_idxb & dontcare_idxb );
    end
    if ~includeProposed
        valid_idxb = valid_idxb & ~proposed_idxb;
    end
    if numMaxProposals < inf
        valid_idxb = valid_idxb & (R.box_ranks <= numMaxProposals);
    end
    
    if ~any(valid_idxb), continue; end
    
    boxes_j    = R.boxes(valid_idxb,:);
    gt_boxes_j = R.boxes(R.is_gt(:,j),:);   
    
    if strcmp( PARAM.result_source, 'average' )
        scores_j = zeros(size(boxes_j,1),size(gt_boxes_j,1));
        for k = 1:size(gt_boxes_j,1)
            scores_j(:,k) = PascalOverlap( gt_boxes_j(k,:), boxes_j );
        end
        scores_j = max(scores_j,[],2);
        
        topOverlap(j) = mean( scores_j );
        for k = 1:length(iouT)
            ACC(k,1,j) = mean( double( scores_j>=iouT(k) ) );
        end
        
    else
        switch PARAM.result_source
            case 'scores'
                scores_j = R.scores(valid_idxb,j);
            case 'oracle'
                scores_j = zeros(size(boxes_j,1),size(gt_boxes_j,1));
                for k = 1:size(gt_boxes_j,1)
                    scores_j(:,k) = PascalOverlap( gt_boxes_j(k,:), boxes_j );
                end
                scores_j = max(scores_j,[],2);
            case 'random'
                scores_j = rand(size(boxes_j,1),1);
            otherwise
                error('Unrecognized result_source');
        end
        chosen_box_idx1 = nms([boxes_j,scores_j],0.3);

        chosen_boxes_j  = boxes_j(chosen_box_idx1,:);
        % chosen_scores_j = scores_j(chosen_box_idx1,:);

        if isempty(gt_boxes_j)
            ov_gt = zeros(size(chosen_boxes_j,1),1);
        else
            ov_gt = zeros(size(chosen_boxes_j,1),size(gt_boxes_j,1));
            for k = 1:size(gt_boxes_j,1)
                ov_gt(:,k) = PascalOverlap( gt_boxes_j(k,:), chosen_boxes_j );
            end
            ov_gt = max(ov_gt,[],2);
        end

        dontcare_boxes_j = R.boxes(R.dontcare(:,j),:);
        if isempty(dontcare_boxes_j)
            ov_dontcare = zeros(size(chosen_boxes_j,1),1);
        else
            ov_dontcare = zeros(size(chosen_boxes_j,1),size(dontcare_boxes_j,1));
            for k = 1:size(dontcare_boxes_j,1)
                ov_dontcare(:,k) = PascalOverlap( dontcare_boxes_j(k,:), chosen_boxes_j );
            end
            ov_dontcare = max(ov_dontcare,[],2);
        end

        top_pos = find( ov_dontcare<=ov_gt, 1 );
        if isempty(top_pos), top_pos = length(ov_gt); end
        topOverlap(j) = max( ov_gt(top_pos) );

        for k = 1:length(iouT)
            gt_rank = find( ov_gt>=iouT(k), 1 );
            if ~isempty(gt_rank)
                num_dontcare_before_rank = sum( ...
                    ov_dontcare(1:gt_rank-1)>=iouT(k) );
                prank = gt_rank - num_dontcare_before_rank;
                ACC(k,rankT>=prank,j) = true;
            end
        end
    end
end

if numPhrases == 0
    ACC = ones( [size(ACC,1),size(ACC,2),1] );
else
    ACC = mean( double(ACC), 3 );
end


