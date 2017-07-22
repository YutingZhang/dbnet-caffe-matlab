classdef nldetTextAugmenter_PredefinedPhrases < handle
    
    properties ( GetAccess = public, SetAccess = protected )
        predefined_phrase_net
    end
    
    methods
        function obj = nldetTextAugmenter_PredefinedPhrases( predefined_phrase_net )
            
            obj.predefined_phrase_net = predefined_phrase_net;
            
        end
        
        function T = image_neg( obj, im )
            
            % im.order
            % im.position
            
            obj.predefined_phrase_net.SetInputData( im.order );
            obj.predefined_phrase_net.Forward();
            T = obj.predefined_phrase_net.GetOutputData();
            T = T{1};
            
        end
    end
end
