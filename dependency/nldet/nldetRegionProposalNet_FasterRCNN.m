classdef nldetRegionProposalNet_FasterRCNN < nldetCaffeBlock 
    
    properties (GetAccess = public, SetAccess = protected)
        
        model
        
        boxes
        scores
        
        final_boxes
        
        im_sizes
        scaled_im_sizes
        
        maximum_boxnum_after_nms
        maximum_boxnum_before_nms
        nms_overlap_threshold
        
    end
    
    properties (Constant, GetAccess = public)
        use_caffe_net = 1;
        default_param = scalar_struct( ...
            'Maximum_BoxNum', 300, ...
            'Maximum_BoxNum_BeforeNMS', Inf, ...
            'NMS_Threshold' ,0.9 );
        extra_net_param = scalar_struct( ...
            'single_batch_inputs',{1} ...
            );
    end
    
    methods
        function obj = nldetRegionProposalNet_FasterRCNN( block, PARAM )
            obj@nldetCaffeBlock( block );
            
            obj.maximum_boxnum_after_nms  = PARAM.Maximum_BoxNum;
            obj.maximum_boxnum_before_nms = PARAM.Maximum_BoxNum_BeforeNMS;
            obj.nms_overlap_threshold     = PARAM.NMS_Threshold;
            
            M = load('/mnt/brain2/scratch/yutingzh/private-homedir/src/faster_rcnn/final_model_zyt.mat');
            M = M.proposal_detection_model;
            obj.model = M;
        end
        
        function SetInputData( obj, conv_feat, im_sizes, scaled_im_sizes )
            obj.SetInputData_(conv_feat);
            obj.im_sizes        = im_sizes;
            obj.scaled_im_sizes = scaled_im_sizes;
        end
        
        function Forward( obj )
            obj.Forward_();
            [box_delta_blob, score_blob] = obj.GetOutputData_();
            % need a for to enumerate different images
            batch_size = size(box_delta_blob,4);
            boxes_cell  = cell(batch_size,1);
            scores_cell = cell(batch_size,1);
            aboxes_cell = cell(batch_size,1);
            for k=1:batch_size
                [boxes_cell{k},scores_cell{k}] = frcnn.rawdelta_to_boxes( ...
                    box_delta_blob(:,:,:,k), score_blob(:,:,:,k), ...
                    obj.im_sizes(k,:), obj.scaled_im_sizes(k,:), obj.model.conf_proposal );
                aboxes_k = frcnn.boxes_filter( [boxes_cell{k},scores_cell{k}], ...
                    obj.maximum_boxnum_before_nms, obj.nms_overlap_threshold, ...
                    obj.maximum_boxnum_after_nms );
                % remark the boxes here are 1-based
                aboxes_cell{k} = bsxcat( 2, k, aboxes_k(:,[2 1 4 3]) ); % [x1,y1,x2,y2] -> [image_id,y1,x1,y2,x2]
            end
            obj.boxes  = boxes_cell;
            obj.scores = scores_cell;
            obj.final_boxes = cat(1,aboxes_cell{:});
        end
        
        function final_boxes = GetOutputData( obj )
            final_boxes = obj.final_boxes;
        end
        
        function SetOutputDiff( obj, varargin )
            % do nothing, because I cannot set diff to them
        end
        
    end
    
end
