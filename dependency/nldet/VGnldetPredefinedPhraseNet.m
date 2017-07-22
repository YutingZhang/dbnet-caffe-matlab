classdef VGnldetPredefinedPhraseNet < handle

    properties ( GetAccess = public, SetAccess = protected )
        image2phrases
        vggl
        cur_image_order
        cur_text
        param
    end
    
    properties ( Constant )
        use_caffe_net = 0;
        default_param = scalar_struct( ...
            'AnnotationName', 'det_images_for_phrases_all' );
    end
    
    methods
        function obj = VGnldetPredefinedPhraseNet( block_def, PARAM )
            
            obj.param = PARAM;
            
            try
                obj.image2phrases = vec(VGAnnotation([obj.param.AnnotationName '_im2ph']));
            catch
                IMG_IND = VGAnnotation('image-ind');
                phrases2images = VGAnnotation(obj.param.AnnotationName);
                obj.image2phrases = vec(nldet_phrase2image_to_image2phrase(phrases2images,IMG_IND));
            end
            
            obj.vggl = VGnldetGTLoader();
            
        end
        
        function SetInputData( obj, image_order )
            obj.cur_image_order = vec(image_order).';
        end
        
        function Forward( obj )
            
            source_ids  = obj.image2phrases( obj.cur_image_order );
            num_phrases = cellfun( @numel, source_ids );
            source_ids  = [source_ids{:}];
            
            phrases = obj.vggl.phrase(source_ids);
            T = struct( 'phrase', vec(phrases), ...
                'source', num2cell(vec(source_ids)) );
            obj.cur_text = mat2cell( T, num_phrases, 1 );
            
        end
        
        function T = GetOutputData( obj )
            
            T = obj.cur_text;
            
        end
        
    end    
    
end
