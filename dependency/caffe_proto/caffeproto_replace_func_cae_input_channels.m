function [X, varargout] = caffeproto_replace_func_cae_input_channels( varargin )


if ischar( varargin{1} )
    if strcmp( varargin{1}, 'extend' )
        X = {};
        return
    elseif strcmp( varargin{1}, 'adjacent' )
        X = [0 1; 0 0];
        %varargout{1} = [1]; % in
        %varargout{2} = [2]; % out
        return;
    end
end

subS = varargin{1};
ARGS = varargin{end};

if strcmp( subS(2).type, 'CAEStack' )
    X = subS;
    if isempty(subS(2).layers)
        storedInputShape = caffeproto_avail_ioshape(subS(1), 't', 1);
    else
        storedInputShape = caffeproto_avail_ioshape(subS(2).layers(end), 'b', 1);
    end
    if ~isempty(storedInputShape)
        X(2).cae_stack_param.input_channels = storedInputShape;
    else
        if isfield( subS(1), 'convolution_param' ) && ...
                isfield(subS(1).convolution_param,'num_output')
            X(2).cae_stack_param.input_channels = subS(1).convolution_param.num_output;
        elseif isfield( subS(1), 'inner_product_param' ) && ...
                isfield(subS(1).inner_product_param,'num_output')
            X(2).cae_stack_param.input_channels = subS(1).inner_product_param.num_output;
        else
            assert( ~isempty(ARGS) && ~isempty(ARGS{1}), ...    % ARGS{1}
                'need to specify the input channel number' );
            X(2).cae_stack_param.input_channels = ARGS{1};
        end
    end
    
    if strcmp( subS(1).type, 'ConvCombo' )
        if ~isempty(subS(1).conv_combo_param.aux_layers) && ...
                ismember( 'ReLU', subS(1).conv_combo_param.aux_layers )
            X(2).cae_stack_param.nonnegative_input = 1;
        else
            X(2).cae_stack_param.nonnegative_input = 0;
        end
    elseif strcmp( subS(1).type, 'ReLU' )
        X(2).cae_stack_param.nonnegative_input = 1;
    else
        assert( numel(ARGS)>=2 && ~isempty(ARGS{2}), ...    % ARGS{1}
            'need to specify whether input is non-negative' );
        X(2).cae_stack_param.nonnegative_input = ARGS{2};
    end
    
    X(1).replace_at = 1;
    X(2).replace_at = 2;
else
    X = [];
end
