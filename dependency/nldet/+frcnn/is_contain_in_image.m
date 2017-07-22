function contained = is_contain_in_image(boxes, im_size)

contained = boxes >= 1 & bsxfun(@le, boxes, [im_size(2), im_size(1), im_size(2), im_size(1)]);
contained = all(contained, 2);
