classdef nldetTextAugmenter_Example < handle
    properties ( Access = private )
        image_sampler
    end
    methods
        function obj = nldetTextAugmenter_Random( image_sampler )
            obj.image_sampler = image_sampler;
        end
        
        function [T, L] = image( obj, im )
            
            % im.order
            % im.position
            
        end
        function T = image_neg( obj, im )
            
            % im.order
            % im.position
            
        end
        function T = image_pos( obj, im )
            
            % im.order
            % im.position
            
        end
        
        function [T, L] = region( obj, im, rgn )
            
            % im.order
            % im.position
            
            % rgn.order
            % rgn.text
            
            
        end
        function T = region_neg( obj, im, rgn )
            
            % im.order
            % im.position
            
            % rgn.order
            % rgn.text
            
            
        end
        function T = region_pos( obj, im, rgn )
            
            % same as region_neg
            
        end
        
        function feedback( obj, im, pair_scores, D )
            
            % im.order
            % im.position
            
            % pair_scores: the standardized pair_scores
            % D is the data block from BatchSampler
            % pair_scores and D are all just for a single image
            
            D.text_pool;
            
        end
        
    end
end
