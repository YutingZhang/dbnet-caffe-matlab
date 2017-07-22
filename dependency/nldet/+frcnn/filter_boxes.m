function [boxes, scores] = filter_boxes(min_box_size, boxes, scores)

widths = boxes(:, 3) - boxes(:, 1) + 1;
heights = boxes(:, 4) - boxes(:, 2) + 1;

valid_ind = widths >= min_box_size & heights >= min_box_size;
boxes = boxes(valid_ind, :);
scores = scores(valid_ind, :);

