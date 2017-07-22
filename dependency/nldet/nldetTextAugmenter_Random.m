classdef nldetTextAugmenter_Random < handle
    
    properties ( Access = private )
        image_sampler
        gt_loader
        phrase_number
        rand_stream
    end
    
    methods
        function obj = nldetTextAugmenter_Random( ...
                image_sampler, gt_loader, phrase_number, rand_seed )
            
            assert(~isempty(gt_loader),'gt_loader shoulde not be empty');
            
            if ~exist('phrase_number', 'var')
                phrase_number = 1;
            end
            if ~exist('rand_seed', 'var')
                rand_seed = 80432;
            end
            
            obj.image_sampler = image_sampler;
            obj.gt_loader     = gt_loader;
            obj.phrase_number = phrase_number;
            obj.rand_stream   = RandStream('mrg32k3a','seed',rand_seed);
            
        end
        
        function obj = set_phrase_number( obj, phrase_number )
            
            obj.phrase_number = phrase_number;
            
        end
        
        function T = image_neg( obj, im )
            
            % im.id
            % im.order
            % im.position
            
            rand_neg_image_orders = obj.image_sampler.random( obj.phrase_number );
            rand_text_pool = cell( length(rand_neg_image_orders), 1 );
            N = length(rand_neg_image_orders);
            rv = obj.rand_stream.rand(N,1);
            for k = 1:N
                A = obj.gt_loader.get( rand_neg_image_orders(k) );
                rand_gt_order = max(1, ceil(rv(k) * numel(A)) );
                rand_text_pool{k} = A(rand_gt_order).text;
            end
            T = cat(1, rand_text_pool{:});
            [~,ia,~] = unique( [T.source] );
            T = T(ia);
        end
        
        function snapshot( obj, output_prefix )
            fn = [output_prefix '.mat'];
            S = scalar_struct( 'phrase_number', obj.phrase_number, ...
                'rand_stream', obj.rand_stream );
            mkpdir_p( output_prefix );
            save_from_struct( fn, S );
        end
        
        function try_restore( obj, input_prefix, no_try )
            
            if nargin<3
                no_try = 0;
            end
            
            fn = [input_prefix '.mat'];
            file_existed = boolean(exist(fn, 'file'));
            if no_try
                assert( file_existed, 'File does not exist' );
            else
                return;
            end
            S = load( fn );
            obj.phrase_number = S.phrase_number;
            obj.rand_stream   = S.rand_stream;

        end
        
        function restore( obj, input_prefix )
            
            obj.try_restore( input_prefix, 1 );
            
        end

    end
    
end
