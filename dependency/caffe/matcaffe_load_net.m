function varargout = matcaffe_load_net( net_path, batch_size, outputFolder_or_Phase, ...
    extra_net_param )
% [net1, net2, ...] = matcaffe_load_net( net_path, [batch_size1, batch_size2, ...], ... 
%                                        outputFolder_or_Phase, extra_net_param )
% use pbEnum to specify phase
% batch_size: num_of_networks * (1 or num_of_input_blob) 

assert( all(batch_size>0) && all(uint64(batch_size(:))==batch_size(:)), 'batch_size must be positive integer' );

if ~exist('extra_net_param','var') || isempty(extra_net_param)
    extra_net_param = struct();
end
Pdef = struct();
Pdef.inputs  = [];
Pdef.outputs = [];
Pdef.dummy_output_loss_weight = []; % can be read number or inf
Pdef.input_force_backward = 0; % can be 1 for all, 0 for none, string cell for particular blobs
Pdef.place_holders = [];

PARAM = xmerge_struct( Pdef, extra_net_param );
if isempty(PARAM.place_holders)
    PARAM.place_holders = struct();
end

output_net_file = 0;
phase = 'dummy';
if exist( 'outputFolder_or_Phase', 'var' ) && ~isempty(outputFolder_or_Phase)
    if isa( outputFolder_or_Phase, 'proto_enum_class' )
        phase = outputFolder_or_Phase.val;
    else
        output_net_file = 1;
    end
end
if output_net_file
    output_folder = outputFolder_or_Phase;
else
    [output_folder, output_folder_cleanup] = get_tmp_folder( 'caffe_load_net' );
end

if ischar( net_path )
    assert( exist( net_path, 'file' )~=0, 'net prototxt does not exist' );
    S1 = prototxt2struct( net_path );
else
    assert( isscalar(net_path) && isstruct(net_path), ...
        'invalid net specification' );
    S1 = net_path;
end

% replace place holder with values
S1 = pbSpecializePlaceHolders( S1, PARAM.place_holders );

% trim network according to outputs and inputs
input_specified  = ~isempty( PARAM.inputs );
output_specified = ~isempty( PARAM.outputs );
if input_specified && output_specified
    [~,S1] = caffeproto_subnet(S1,'range',PARAM.inputs, PARAM.outputs);
elseif input_specified
    [~,S1] = caffeproto_subnet(S1,'head',PARAM.inputs);    
elseif output_specified
    [~,S1] = caffeproto_subnet(S1,'tail',PARAM.outputs);
end

% set up dummy loss weight to force back-propagation
if ~isempty( PARAM.dummy_output_loss_weight )
    A1 = caffeproto_get_aux(S1);
    outputLayerBlobIdx = A1.layer( A1.topLayerIdx ).preBlobs;
    for k = 1:numel(outputLayerBlobIdx)
        lb = outputLayerBlobIdx{k};
        for j = 1:size(lb,2)
            layerId = lb(1,j);
            blobId  = lb(2,j);
            num_layer_tops = numel(S1.layer(layerId).top);
            if ~isfield( S1.layer(layerId), 'loss_weight' ) || ...
                    ( num_layer_tops>0 && isempty(S1.layer(layerId).loss_weight) )
                S1.layer(layerId).loss_weight = zeros(1, num_layer_tops );
            end
            is_disabled_blob = 0;
            if output_specified
                if ~ismember( S1.layer(layerId).top{blobId}, PARAM.outputs )
                    % disable back-propgateion if not in outputs
                    S1.layer(layerId).loss_weight(blobId) = 0;
                    is_disabled_blob = 1;
                end
            end
            if ~is_disabled_blob
                if S1.layer(layerId).loss_weight(blobId) == 0
                    S1.layer(layerId).loss_weight(blobId) = PARAM.dummy_output_loss_weight;
                end
            end
        end
    end
end

% in order to get input blob size
S0 = caffeproto_add_ioshape( S1 );

% generate multiple nets
numNet = size(batch_size,1);

varargout = cell(1,numNet);

for k = 1:numNet
    S = S0;
    S = caffeproto_make_deploy( S, batch_size(k,:), PARAM.input_force_backward );
    S = caffeproto_replace( S, 'clear_aux' );
    revised_net_path = fullfile( output_folder, sprintf('net%d.prototxt',k) );
    struct2prototxt( S, revised_net_path );
    if output_net_file
        varargout{k} = revised_net_path;
    else
        caffe_net = caffe.Net( revised_net_path, phase );
        varargout{k} = matcaffeNet( caffe_net, true );
    end
end


