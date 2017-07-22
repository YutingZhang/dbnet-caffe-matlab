classdef nldetTextFeatNet < nldetCaffeBlock 

    properties (GetAccess = public, SetAccess = protected)
        
        encoder_constructor
        encoder
        
        param
        
    end
    
    methods
        
        function obj = nldetTextFeatNet( block, PARAM )
            
            obj@nldetCaffeBlock(block);
            
            extra_param_path = fullfile( fileparts( ...
                block.aux.block_def.net_path ), 'param.prototxt' );
            extra_param = prototxt2struct( extra_param_path, [], 0 );
            obj.param = xmerge_struct( extra_param, PARAM );
            
            obj.encoder_constructor = eval( ...
                sprintf( '@(varargin) nldetTextEncoder_%s( varargin{:} )', ...
                obj.param.encoder_type ) );
            enc_param = obj.param.encoder_param;
            obj.encoder = obj.encoder_constructor( enc_param );
            
        end

        function SetInputData( obj, phrases, varargin )
            % convert text to encoding
            % feed to the network
            
            phrase_blob = obj.encoder.encode( phrases );
            obj.SetInputData_( phrase_blob, varargin{:} );
            
        end
        
        function UseDeviceMat( obj, varargin )
            obj.UseDeviceMat_( varargin{:} );
            if isprop(obj.encoder,'use_gpu')
                obj.encoder.use_gpu = obj.block.use_device;
            end
        end
        
        function UseCPUMat( obj )
            obj.UseCPUMat_();
            obj.encoder.use_gpu = obj.block.use_device;
        end
        
        
    end

end
