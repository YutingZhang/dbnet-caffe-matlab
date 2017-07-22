function boxes = clip_boxes(boxes, im_width, im_height)

% x1 >= 1 & <= im_width
boxes(:, 1:4:end) = max(min(boxes(:, 1:4:end), im_width), 1);
% y1 >= 1 & <= im_height
boxes(:, 2:4:end) = max(min(boxes(:, 2:4:end), im_height), 1);
% x2 >= 1 & <= im_width
boxes(:, 3:4:end) = max(min(boxes(:, 3:4:end), im_width), 1);
% y2 >= 1 & <= im_height
boxes(:, 4:4:end) = max(min(boxes(:, 4:4:end), im_height), 1);
