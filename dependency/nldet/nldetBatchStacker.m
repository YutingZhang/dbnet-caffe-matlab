classdef nldetBatchStacker < handle

    properties (Access = protected)
        imgs
        im_blob
        trans_blob
        model
    end
    
    methods
        
        function obj = nldetBatchStacker( StageSpecificDir4BatchStacker )
            M = load(fullfile(StageSpecificDir4BatchStacker,'model.mat'));
            M = M.proposal_detection_model;
            obj.model = M;
        end
        
        function SetInputData( obj, imgs )
            obj.imgs = imgs;
        end
        
        function Forward( obj )
            conf = obj.model.conf_proposal;
            im_blob_cell = cell(1,1,1,numel(obj.imgs));
            trans_blob_cell = cell(1,1,numel(obj.imgs));
            for j = 1:numel(obj.imgs)
                im = obj.imgs{j};
                [im_blob_j, im_scale_j] = frcnn.get_image_blob(conf, im);
                
                trans_blob_cell{j} = [eye(2)*im_scale_j,zeros(2,1)];

                % permute data into caffe c++ memory, thus [num, channels, height, width]
                im_blob_j = im_blob_j(:, :, [3, 2, 1], :); % from rgb to brg
                im_blob_j = permute(im_blob_j, [2, 1, 3, 4]);
                im_blob_j = single(im_blob_j);
                im_blob_cell{j} = im_blob_j;
            end
            obj.im_blob = frcnn.im_list_to_blob(im_blob_cell);
            obj.trans_blob = cell2mat(trans_blob_cell);
        end
        
        function [im_blob, trans_blob] = GetOutputData( obj )
            im_blob    = obj.im_blob;
            trans_blob = obj.trans_blob;
        end
        
    end
    
end
