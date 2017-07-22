classdef nldetRegionFeatNet < nldetCaffeBlock 
    
    properties (GetAccess = public, SetAccess = protected)
        
        param
        
    end
    
    properties (Constant, GetAccess = public)
        use_caffe_net = 1;
        extra_net_param = scalar_struct( ...
            'single_batch_inputs',{1} ...
            );
    end
    
    methods
        function obj = nldetRegionFeatNet( block, PARAM )
            obj@nldetCaffeBlock( block );
            
            extra_param_path = fullfile(fileparts( ...
                block.aux.block_def.net_path ), 'param.prototxt' );
            extra_param = prototxt2struct( extra_param_path, [], 0 );
            obj.param   = xmerge_struct( extra_param, PARAM );
        end
        
        function SetInputData( obj, conv_feat, mat_boxes )
            
            switch obj.param.coordinate_type
                case 'faster-rcnn'
                    % [image_id,y1,x1,y2,x2] --> [image_id,x1,y1,x2,y2]
                    % 1-base to 0-base
                    roi_blob = mat_boxes(:,[1 3 2 5 4]) - 1;
                    roi_blob = permute(roi_blob, [3, 4, 2, 1]);
                otherwise
                    error( 'invalid BoxCoordinate_Type' );
            end
            
            obj.SetInputData_( conv_feat, roi_blob );
        end
        
        function [conv_feat_diff, mat_boxes_diff] = GetInputDiff( obj )
            [conv_feat_diff, roi_blob_diff] = obj.GetInputDiff_();
            if isempty(roi_blob_diff)
                mat_boxes_diff = roi_blob_diff;
            else
                switch obj.param.coordinate_type
                    case 'faster-rcnn'
                        mat_boxes_diff = permute(roi_blob_diff, [4,3,1,2]);
                        mat_boxes_diff = mat_boxes_diff(:,[1 3 2 5 4]);
                    otherwise
                        error( 'invalid BoxCoordinate_Type' );
                end
            end
        end
        
    end
    
end
