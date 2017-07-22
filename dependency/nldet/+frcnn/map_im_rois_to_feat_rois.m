function [feat_rois, levels] = map_im_rois_to_feat_rois(conf, im_rois, scales)

im_rois = single(im_rois);

if length(scales) > 1
    widths = im_rois(:, 3) - im_rois(:, 1) + 1;
    heights = im_rois(:, 4) - im_rois(:, 2) + 1;

    areas = widths .* heights;
    scaled_areas = bsxfun(@times, areas(:), scales(:)'.^2);
    levels = max(abs(scaled_areas - 224.^2), 2); 
else
    levels = ones(size(im_rois, 1), 1);
end

feat_rois = round(bsxfun(@times, im_rois-1, scales(levels)) / conf.feat_stride) + 1;
