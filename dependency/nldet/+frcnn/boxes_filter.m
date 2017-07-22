function aboxes = boxes_filter(aboxes, pre_nms_topN, nms_overlap_thres, after_nms_topN)
    % to speed up nms
    if pre_nms_topN > 0
        aboxes = aboxes(1:min(length(aboxes), pre_nms_topN), :);
    end
    % do nms
    if nms_overlap_thres > 0 && nms_overlap_thres < 1
        aboxes = aboxes(nms(aboxes, nms_overlap_thres), :);       
    end
    if after_nms_topN > 0
        aboxes = aboxes(1:min(size(aboxes,1), after_nms_topN), :);
    end
end
