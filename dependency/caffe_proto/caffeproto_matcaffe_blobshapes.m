function [blobShapes,blobNames] = caffeproto_matcaffe_blobshapes( S0 )

TMP_DIR  = '/tmp';
TMP_FN   = ['caffeproto_add_ioshape' datestr(now,30) int2str(randi(1e8)) '.prototxt'];
TMP_PATH = fullfile(TMP_DIR,TMP_FN);

struct2prototxt(S0,TMP_PATH);

net = caffe.Net(TMP_PATH,'dummy');

if exist( 'remove_file', 'file' ) == 3
    remove_file(TMP_PATH);
else
    delete(TMP_PATH);
end

cleanupObj = onCleanup( @() net.reset() );

blobNames  = net.blob_names;
blobN = length(blobNames);
blobShapes = cell(1,blobN+1);
for k = 1:blobN
   B = net.blobs(blobNames{k});
   bs = B.shape;
   blobShapes{k} = bs(end-1:-1:1);
end
blobShapes{end} = [];

