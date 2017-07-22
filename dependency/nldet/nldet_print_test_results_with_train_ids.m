function varargout = nldet_print_test_results_with_train_ids( ...
    train_ids, test_setting, num_boxes, iou_th, train_titles )



if ~exist('iou_th','var')
    iou_th = [];
end

test_ids = nldet_lastest_test_ids(train_ids,test_setting);

if ~exist('train_titles','var') || isempty(train_titles)
    train_titles = arrayfun( @(train_id,test_id) ...
        sprintf('train-%d;test-%d', train_id, test_id), ...
        vec(train_ids), vec(test_ids), ...
        'UniformOutput', 0 );
else
    train_titles = arrayfun( @(train_id,test_id,title_str) ...
        sprintf('train-%d;test-%d: %s', train_id, test_id, title_str{1}), ...
        vec(train_ids), vec(test_ids), vec(train_titles), ...
        'UniformOutput', 0 );    
end

varargout = cell(1,nargout);
[varargout{:}] = nldet_print_test_results( test_ids, num_boxes, ...
    iou_th, train_titles );

