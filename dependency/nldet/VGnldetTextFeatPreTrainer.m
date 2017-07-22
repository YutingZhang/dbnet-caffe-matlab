classdef VGnldetTextFeatPreTrainer < handle
    
    properties (GetAccess=public, SetAccess=protected)
        tfn
        vocab
        phrases
        batch_size
        vocab_size
        sampler
        block_def
        
        train_state
    end
    
    methods (Access = public) 
        function obj = VGnldetTextFeatPreTrainer( tfn, varargin )
            obj.tfn = tfn;
            obj.vocab = VGAnnotation('vocabulary');
            obj.phrases = VGAnnotation('phrases_in_words');
            
            input_label_shape = tfn.block.net.blobs( tfn.block.net.inputs{2} ).shape();
            
            obj.vocab_size = input_label_shape(1);
            obj.batch_size = input_label_shape(4);
            
            obj.block_def = varargin{1};
            SpecificDir4PrepDataset = obj.block_def.SPECIFIC_DIRS.PrepDataset;
            
            meta_cache_path = fullfile( SpecificDir4PrepDataset, 'meta.v7.mat' );
            S = cached_file_func( @load7, meta_cache_path, 'vg-cache', 20 );
            subsetImageIndexes = cell(1,numel(obj.block_def.DataSubset_Name));
            for k = 1:length(subsetImageIndexes)
                subsetImageIndexes{k} = subsref( S, substruct('.',[obj.block_def.DataSubset_Name{k} '_idxb']) );
            end
            subsetImageIndexes = cat(3,subsetImageIndexes{:});
            subsetImageIndex = any(subsetImageIndexes,3);
            
            obj.sampler = IndexSampler( subsetImageIndex, scalar_struct('shuffle',1) );
            
            obj.train_state = struct();
            obj.train_state.iter = 0;
        end
        function step( obj )
            I = zeros(obj.batch_size,1);
            for k=1:obj.batch_size
                I(k) = obj.sampler.next();
            end
            P = cell(obj.batch_size,1);
            L = zeros(obj.vocab_size,obj.batch_size,'single');
            for k=1:obj.batch_size
                u = I(k);
                P{k} = merge_into_single_phrase( obj.phrases{u}, obj.vocab.dict );
                active_word_idx = unique( obj.phrases{u} );
                active_word_idx(active_word_idx>obj.vocab_size) = [];
                L(active_word_idx,k) = 1;
            end
            L = reshape(L,[obj.vocab_size,1,1,obj.batch_size]);
            obj.tfn.SetInputData(P,L);
            obj.tfn.ForwardBackward();
            obj.tfn.Update();
            obj.train_state.iter = obj.train_state.iter + 1;
            [loss_pos,loss_neg] = obj.tfn.GetOutputData();
            fprintf( 'Iter %d : \tloss(pos): %g\tloss(neg): %g\n', ...
                obj.train_state.iter, loss_pos, loss_neg );
        end
    end
    
end
