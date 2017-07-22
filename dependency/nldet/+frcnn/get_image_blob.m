function [blob, im_scales] = get_image_blob(conf, im)

if length(conf.test_scales) == 1
    [blob, im_scales] = frcnn.prep_im_for_blob(im, conf.image_means, conf.test_scales, conf.test_max_size);
else
    [ims, im_scales] = arrayfun(@(x) frcnn.prep_im_for_blob(im, conf.image_means, x, conf.test_max_size), conf.test_scales, 'UniformOutput', false);
    im_scales = cell2mat(im_scales);
    blob = frcnn.im_list_to_blob(ims);    
end

