function P = nldetBlockInitParamFromDir( dir_path, phase_str )

if nargin<2 || isempty( phase_str )
    phase_str = 'test';
end

P = struct();
P.external   = 0;
P.mclass     = [];
P.trainer_mclass = [];
P.seal_caffe_net = 0;
P.net_path   = [];
P.model_path = [];
P.outputs    = [];
P.outputs_backward = [];

if exist(fullfile( dir_path, 'external' ),'file')
    % DEPRECATED: should not use
    P.external   = 1;
    return;
end

net_path = fullfile( dir_path, 'net.prototxt' );
use_caffe_net = boolean( exist(net_path, 'file') );

mclass_path = fullfile( dir_path, 'mclass.txt' );
if exist(mclass_path,'file')
    mclass_name = readlines( mclass_path, 1 );
    P.mclass = nldetMClassAlias( mclass_name{1} );
    use_caffe_net = default_eval( sprintf( '%s.use_caffe_net', P.mclass ), use_caffe_net);
    if use_caffe_net
        P.seal_caffe_net = default_eval( sprintf( '%s.seal_caffe_net', P.mclass ), 0);
    end
    if numel(mclass_name)>=2
        P.trainer_mclass = mclass_name{2};
    end
end

if use_caffe_net

    P.net_path = net_path;

    model_path = fullfile( dir_path, 'init.caffemodel' );
    if exist(model_path, 'file')
        P.model_path = model_path;
    end

end

output_blob_path = fullfile( dir_path, sprintf('output_blob_%s.txt', phase_str) );
if ~exist(output_blob_path, 'file')
    output_blob_path = fullfile( dir_path, 'output_blob.txt' );
end
if exist(output_blob_path,'file')
    outputs = cached_file_func( @(a) readlines(a,1), ...
        output_blob_path, 'block_init:output_blob', 200 );
    outputs = reshape( outputs, 1, numel(outputs) );
    outputs_backward = cellfun( @(a) a(1)~='>', outputs );
    outputs(~outputs_backward) = cellfun( @(a) a(2:end), ...
        outputs(~outputs_backward), 'UniformOutput', 0 );
    % remark: > for no backwards
else
    if use_caffe_net
        S = prototxt2struct( P.net_path );
        A = caffeproto_get_aux(S);
        tpb = A.layer(A.topLayerIdx).preBlobs;
        outputs = cell( 1, numel( tpb ) );
        for k = 1:numel(outputs)
            layerId = tpb{k}(1,1);
            blobId  = tpb{k}(2,1);
            outputs{k} = S.layer(layerId).top{blobId};
        end
        outputs_backward = true( size(outputs) );
    else
        outputs = [];
        outputs_backward = [];
    end
end
P.outputs = outputs;
P.outputs_backward = outputs_backward;

