classdef nldetConvFeatNet < handle

    properties (GetAccess = public, SetAccess = private)
        
        block
        
    end
    
    methods
        function obj = nldetConvFeatNet( block )
            obj.block = block;
        end
        
        function SetInputData( obj, img_data )
            obj.block.SetInputData( img_data );
        end
        
        function F = GetOutputData( obj )
            F = obj.block.GetOutputData();
        end
        
        function Forward( obj )
            obj.block.Forward();
        end
        
        function SetOutputDiff( obj, conv_feat_diff )
            obj.block.SetOutputDiff( conv_feat_diff );
        end
        
        function Backward( obj )
            obj.block.Backward();
        end
        
        function Update( obj )
            obj.block.Update();
        end
        
    end
    
end
