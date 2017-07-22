function C = nldetGetDataLoaderConstructor( SpecificDir4PrepDataset )

if ~exist('SpecificDir4PrepDataset','var')
    SpecificDir4PrepDataset = evalin( 'caller', 'SPECIFIC_DIRS.PrepDataset' );
end

data_loader_config = prototxt2struct( fullfile( SpecificDir4PrepDataset, ...
    'data_loader.prototxt' ), [], 0 );

C = struct();

C.image_sampler = nldetMClass( data_loader_config.image_sampler );
C.gt_loader     = nldetMClass( data_loader_config.gt_loader );

