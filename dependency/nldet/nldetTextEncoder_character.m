classdef nldetTextEncoder_character < handle
    
    properties (GetAccess=public, SetAccess=protected)
        param
        codebook
        gpu_vocab
    end
    properties (Access=public)
        use_gpu = false
    end
    
    methods
        function obj = nldetTextEncoder_character( PARAM )
            
            % parameter
            obj.param = PARAM;
            if isfield(obj.param, 'input_length')
                if isinf( obj.param.input_length )
                    obj.param.input_length = [];
                end
            else
                obj.param.input_length = [];
            end
            
            % code book
            %  -- load
            cb = obj.param.codebook;
            if isa( cb, 'proto_enum_class' )
                cb = fullfile( fileparts( mfilename( 'fullpath' ) ), ...
                    'nldetTextEncoder_character.vocab.d', ...
                    sprintf('%s.nldetTextEncoder_character.vocab.mat', cb.val() ) );
            else
                if iscell(cb) && isscalar(cb) && ischar(cb{1})
                    cb = cb{1};
                else
                    error( 'Invalid codebook specification' );
                end
            end
            assert( boolean(exist(cb,'file')), 'codebook file does not exist' );
            
            S = cached_file_func( @load, cb, 'nldet-text-codebook', 20 );
            % S = loaded2single_var(S);
            obj.codebook = S;
            
            %  -- canonicalize
            if isfield(obj.codebook, 'preprocess_func')
                if ischar(obj.codebook.preprocess_func)
                    obj.codebook.preprocess_func = eval( ...
                        obj.codebook.preprocess_func );
                end
            else
                obj.codebook.preprocess_func = [];
            end
        end
        
        function C = encode( obj, T )
            if ischar(T)
                T = {T};
            end
            
            if ~isempty(obj.codebook.preprocess_func)
                T = cellfun( obj.codebook.preprocess_func, T, 'UniformOutput', 0 );
            end
            
            if obj.use_gpu
                if isempty(obj.gpu_vocab)
                    vocab = gpuArray( uint8(obj.codebook.vocab) );
                    obj.gpu_vocab = vocab;
                else
                    vocab = obj.gpu_vocab;
                end
            else
                vocab = obj.codebook.vocab;
            end
            
            C = character_enc( T, vocab, [], ...
                obj.param.input_length );
            
        end
        
    end
    
end
