function A = matcaffe_net_statistics( net )

A = struct();
for k = 1:numel(net.layer_names)
    A.layer(k).name = net.layer_names{k};
    L = net.layers(A.layer(k).name);
    for j = 1:numel(L.params)
        A.layer(k).param(j) = matcaffe_blob_statistics( L.params(j) );
    end
end

for k = 1:numel(net.blob_names)
    A.blob(k).name = net.blob_names{k};
    B = net.blobs(A.blob(k).name);
    A.blob(k).stat = matcaffe_blob_statistics( B );
end

function A = matcaffe_blob_statistics( B )

A=struct();
A.shape = B.shape();

D = B.get_data();
D = vec(D);
non_zero_idxb = D~=0;
num_non_zero = sum(non_zero_idxb);
A.non_zero    = num_non_zero/numel(D);

positive_idxb = D>0;
num_positive  = sum(positive_idxb);
A.positive    = num_positive/numel(D);

if numel(D)>=2
    A.zero_normal_sigma = sqrt(sum(D.*D)/(numel(D)-1));
else
    A.zero_normal_sigma = [];
end


if numel(D)>=2
    p = fitdist(D,'Normal');
    A.normal_mu    = p.mu;
    A.normal_sigma = p.sigma;
else
    A.normal_mu    = [];
    A.normal_sigma = [];
end

if num_positive>=2
    p = fitdist(D(positive_idxb),'HalfNormal');
    A.half_normal_sigma = p.sigma;
else
    A.half_normal_sigma = [];
end

