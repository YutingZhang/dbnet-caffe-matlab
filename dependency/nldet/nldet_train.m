function M = nldet_train( sampler, PARAM )
% train nldet model
% 
% a nldet model is composed of 
%    convolutional feature extraction network
%    region feature extraction network
%    text feature extraction network
%    text-region compatibility network
%
% sampler.image
% sampler.region
% sampler.text
% sampler.pair

% default params
Pdef = struct();
Pdef.max_epoch = 10;
Pdef.cache_dir = '/tmp/nldet_train';
Pdef.save_when_interrupted = 1;
Pdef.num_image_per_iter = 2;

PARAM = xmerge_struct(Pdef,PARAM);

% training iterations
while sampler.image.epoch() <= PARAM.max_epoch
    D = repmat(struct(),PARAM.num_image_per_iter,1);
    for j = 1:PARAM.num_image_per_iter
        % sample an image
        D(j).sampleId = sampler.next();
        D(j).im = sampler.image_meta();
        D(j).I  = sampler.image();
        % sample regions and text
        sampler.ann.set_id(D(j).sampleId);
    end
    % region proposal (can be back-propagatable)
end
