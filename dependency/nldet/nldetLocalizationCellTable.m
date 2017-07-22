function T = nldetLocalizationCellTable( ACC, overlap_threshold, ranking_threshold )

if isstruct(ACC)
    T = structfun( @(a) nldetLocalizationCellTable(a, ...
        overlap_threshold, ranking_threshold ), ACC, 'UniformOutput', 0 );
    return;
end

rowTitles = arrayfun( @(a) sprintf('IoU >= %g', a), vec(overlap_threshold), ...
    'UniformOutput', 0 );

colTitles = arrayfun( @(a) sprintf('Rank %d', a), vec(ranking_threshold).', ...
    'UniformOutput', 0 );

T = arrayfun( @(a) sprintf('%5s',num2str(a,'%.2f')), ACC*100, ...
    'UniformOutput', 0 );
T = [{'Accuracy / %'}, colTitles;
    rowTitles, T];

