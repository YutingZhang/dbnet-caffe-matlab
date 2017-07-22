classdef VGnldetTextAugmenter_PredefinedPhrases

    properties ( Access = private )
        image2phrases;
        vggl
    end
    
    methods
        function obj = VGnldetTextAugmenter_PredefinedPhrases( annotation_fn )
            
            IMG_IND = VGAnnotation('image-ind');
            I = codebook_cols( IMG_IND, {'image_id','image_order'} );
            obj.image2phrases = cell(size(I,1),1);
            
            phrases2images = VGAnnotation(annotation_fn);
            
            t1 = tic_print( 'Build image2phrases index' );
            n  = size(phrases2images,2);
            for k = 1:n
                
                phrase_id  = phrases2images{1,k};
                image_info = phrases2images{2,k};
                
                image_ids  = [image_info.pos,image_info.neg];
                image_orders = map_with_codebook( image_ids, I );
                
                for j = 1:numel(image_orders)
                    obj.image2phrases{image_orders(j)} = [ ...
                        obj.image2phrases{image_orders(j)}, phrase_id ];
                end
                
            end
            toc_print(t1);
            
            obj.vggl = VGnldetGTLoader();
            
        end
        
        function T = image_neg( obj, im )
            
            % im.order
            % im.position
            
            source_ids = obj.image2phrases{ im.order };
            phrases = obj.vggl.phrase(source_ids);
            
            T = struct( 'phrase', phrases, ...
                'source', num2cell(source_ids) );
            
        end
        
    end    
    
end
