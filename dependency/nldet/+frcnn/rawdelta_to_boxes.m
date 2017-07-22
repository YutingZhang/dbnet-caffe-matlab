function [pred_boxes, scores, box_deltas_, anchors_, scores_] = ...
    rawdelta_to_boxes( box_delta_blob, score_blob, im_size, scaled_im_size, conf )

box_deltas = box_delta_blob;
% Apply bounding-box regression deltas
featuremap_size = [size(box_deltas, 2), size(box_deltas, 1)];
% permute from [width, height, channel] to [channel, height, width], where channel is the
    % fastest dimension
box_deltas = permute(box_deltas, [3, 2, 1]);
box_deltas = reshape(box_deltas, 4, [])';

anchors = frcnn.proposal_locate_anchors(conf, im_size, conf.test_scales, featuremap_size);
pred_boxes = frcnn.fast_rcnn_bbox_transform_inv(anchors, box_deltas);
  % scale back
pred_boxes = bsxfun(@times, pred_boxes - 1, ...
    ([im_size(2), im_size(1), im_size(2), im_size(1)] - 1) ./ ([scaled_im_size(2), scaled_im_size(1), scaled_im_size(2), scaled_im_size(1)] - 1)) + 1;
pred_boxes = frcnn.clip_boxes(pred_boxes, im_size(2), im_size(1));

assert(conf.test_binary == false);
% use softmax estimated probabilities
scores = score_blob(:, :, end);
scores = reshape(scores, size(box_delta_blob, 1), size(box_delta_blob, 2), []);
% permute from [width, height, channel] to [channel, height, width], where channel is the
    % fastest dimension
scores = permute(scores, [3, 2, 1]);
scores = scores(:);

box_deltas_ = box_deltas;
anchors_ = anchors;
scores_ = scores;

if conf.test_drop_boxes_runoff_image
    contained_in_image = is_contain_in_image(anchors, round(im_size * im_scales));
    pred_boxes = pred_boxes(contained_in_image, :);
    scores = scores(contained_in_image, :);
end

% drop too small boxes
[pred_boxes, scores] = frcnn.filter_boxes(conf.test_min_box_size, pred_boxes, scores);

% sort
[scores, scores_ind] = sort(scores, 'descend');
pred_boxes = pred_boxes(scores_ind, :);

