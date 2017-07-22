function nldet_localization_ranking_curves( test_ids, test_titles, ...
    num_boxes, iou_th, sim_th, valgin, lstyle )

assert( numel(test_ids)==numel(test_titles), ...
    'the number of test_ids should be the same as the number of test_titles.' );

if ~exist('valgin','var') || isempty(valgin)
    valgin = repmat({'middle'},size(test_ids));
end

if ~exist('lstyle','var') || isempty(lstyle)
    lstyle = repmat({},size(test_ids));
end

switch_pipeline_quiet nldet


test_ids    = vec(test_ids);
test_titles = vec(test_titles);

if sim_th==1
    subfolder_name = 'localization';
else
    subfolder_name = sprintf('localization-sim_%g', sim_th);
end
accuracy_mat_fn = fullfile( subfolder_name, ...
    sprintf('boxes_%d',num_boxes), 'accuracy.mat' );

RANK = cell(length(test_ids),1);
ACC  = cell(length(test_ids),1);
for k = 1:length(test_ids)
    base_folder = fullfile( sysStageDir('Test'), sprintf('no-%d',test_ids(k)) );
    accuracy_path = fullfile(base_folder,accuracy_mat_fn);
    R = load( accuracy_path );
    [~,iou_pos] = ismembertol( iou_th, R.base_loc_param.overlap_threshold );
    ACC{k}  = R.ACC.proposed( iou_pos, :) * 100;
    RANK{k} = R.base_loc_param.ranking_threshold;
end


clf

%minY = min(cellfun(@min, ACC));
maxY = max(cellfun(@max, ACC));
if maxY<25
    limY = ceil( maxY/5 )*5;
else
    limY = ceil( maxY/10 )*10;
end

minX = min(cellfun(@min, RANK));
maxX = max(cellfun(@max, RANK));
lenX = maxX-minX;
if 0
    hold all
    set(gca, 'XLim', [minX-0.5, maxX + lenX*0.3], 'XTick', minX:maxX, ...
        'YLim', [0 limY+limY*0.1] );
    set(gca, 'Box', 'off');
    set(gcf, 'Color', 'white', 'InvertHardcopy', 'off', ...
        'PaperPositionMode', 'auto' );
    xlabel('Rank', 'FontSize', 10);
    ylabel('Recall / %', 'FontSize', 10);
    C = lines(length(test_ids));
    for k = 1:length(test_ids)
        plot( RANK{k}, ACC{k}, lstyle{k}{:}, 'Color', C(k,:), 'LineWidth', 1.5 );
    %    plot( RANK{k}, ACC{k}, 'o-', 'Color', C(k,:), 'LineWidth', 1.5, ...
    %        'MarkerFaceColor', C(k,:), 'MarkerSize', 2 );
        text( RANK{k}(end)+0.05*lenX, ACC{k}(end), test_titles{k}, ...
            'HorizontalAlignment', 'left', 'VerticalAlignment', valgin{k} );
    %    text( RANK{k}(end), ACC{k}(end)+limY*0.03, test_titles{k}, ...
    %        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom' );
    end
    tfig = gcf;
    tfig.Position(3:4) = [380,300]*0.7;
    taxis = gca;
    taxis.Position(3:4) = [0.65, 0.73];
    taxis.FontSize = 9;
else
    hold all
    set(gca, 'XLim', [minX-0.5, maxX + 0.5], 'XTick', minX:maxX, ...
        'YLim', [0 limY] );
    set(gca, 'Box', 'off');
    set(gcf, 'Color', 'white', 'InvertHardcopy', 'off', ...
        'PaperPositionMode', 'auto' );
    xlabel('Rank', 'FontSize', 10);
    ylabel('Recall / %', 'FontSize', 10);
    C = lines(length(test_ids));
    H = [];
    for k = 1:length(test_ids)
        H(k) = plot( RANK{k}, ACC{k}, lstyle{k}{:}, 'Color', C(mod(k,length(test_ids))+1,:), 'LineWidth', 1.5 );
        set(H(k),'DisplayName',test_titles{k});
    %    plot( RANK{k}, ACC{k}, 'o-', 'Color', C(k,:), 'LineWidth', 1.5, ...
    %        'MarkerFaceColor', C(k,:), 'MarkerSize', 2 );
        %text( RANK{k}(end)+0.05*lenX, ACC{k}(end), test_titles{k}, ...
        %    'HorizontalAlignment', 'left', 'VerticalAlignment', valgin{k} );
    %    text( RANK{k}(end), ACC{k}(end)+limY*0.03, test_titles{k}, ...
    %        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom' );
    end
    taxis = gca;
    taxis.FontSize = 9;
    legend( H, 'Location', 'Best' );
end

hold off
