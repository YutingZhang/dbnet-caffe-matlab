classdef nldetRegionProposalNet_Empty < handle
    
    properties (GetAccess = public, SetAccess = protected)
    end
    
    properties ( Constant, GetAccess = public )
        use_caffe_net = 0
    end
    
    methods
        function obj = nldetRegionProposalNet_Empty( block_def, PARAM )
                        
        end
        
        function SetInputData( obj, varargin )
            % do nothing
        end
        
        function Forward( obj )
            % do nothing
        end
        
        function boxes = GetOutputData( obj )
            boxes = zeros(0,5);
        end

        function SetOutputDiff( obj, boxes_diff )
            % do nothing
        end
        
        function Backward( obj )
            % do nothing
        end
        
        function Update( obj )
            % do nothing
        end

        
    end
    
end
