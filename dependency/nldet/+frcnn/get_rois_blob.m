function [rois_blob] = get_rois_blob(conf, im_rois, im_scale_factors)

[feat_rois, levels] = map_im_rois_to_feat_rois(conf, im_rois, im_scale_factors);
rois_blob = single([levels, feat_rois]);
