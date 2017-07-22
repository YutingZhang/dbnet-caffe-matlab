classdef VGImageSampler < IndexSamplerWithDataLoader
    properties ( GetAccess = public, SetAccess = protected )
        img_data
        img_dir
    end
    methods
        function obj = VGImageSampler( subsetImageIndexes, PARAM, SpecificDir4PrepDataset )
            % VGImageSampler( subsetImageIndexes, PARAM [, SpecificDir4PrepDataset] )
            % subsetImageIndexes: 'train', 'val', 'test', 
            %                      logical or numerical indexes. 
            %                      or cell array of any them
            if nargin<2, PARAM = struct(); end
            IMG_DATA = VGAnnotation('image_data');
            assert(isvector(IMG_DATA) && isstruct(IMG_DATA),'IMG_DATA should be a struct vector');
            if ~iscell(subsetImageIndexes)
                subsetImageIndexes = {subsetImageIndexes};
            end
            chosenImageIndexes = false( size(IMG_DATA) );
            for k = 1:length(subsetImageIndexes)
                if ischar(subsetImageIndexes{k})
                    if ~exist('SpecificDir4PrepDataset','var')
                        SpecificDir4PrepDataset = evalin( 'caller', 'SPECIFIC_DIRS.PrepDataset' );
                    end
                    meta_cache_path = fullfile( SpecificDir4PrepDataset, 'meta.v7.mat' );
                    S = cached_file_func( @load7, meta_cache_path, 'vg-cache', 20 );
                    subsetImageIndexes{k} = subsref( S, substruct('.',[subsetImageIndexes{k} '_idxb']) );
                end
                if islogical(subsetImageIndexes{k})
                    assert(isvector(subsetImageIndexes{k}), 'chosenImageIndexes should be a vector');
                    assert( numel(subsetImageIndexes{k}) == numel(IMG_DATA), ...
                        'If chosenImageIndexes is logical, it should have the same number of elememts as IMG_DATA' );
                    chosenImageIndexes = chosenImageIndexes | subsetImageIndexes{k};
                elseif isnumeric(subsetImageIndexes{k})
                    assert(isvector(subsetImageIndexes{k}), 'chosenImageIndexes should be a vector');
                    chosenImageIndexes( subsetImageIndexes{k} ) = true;
                else
                    error('Unrecognized subsetImageIndexes');
                end
            end
            obj = obj@IndexSamplerWithDataLoader(chosenImageIndexes, PARAM);
            obj.img_data = IMG_DATA;
            GS = load_global_settings;
            obj.img_dir = GS.VG_IMAGE_PATH;
            
            obj.set_pre_loader_func( @(i) obj.image_path_at(i) );
            obj.set_loader_func( @imread );

        end
        
        function local_file_path = image_path( obj )
            local_file_path = obj.image_path_at( obj.current() );
        end
        
        function local_file_path = image_path_at( obj, id )
            M = obj.img_data(id);
            local_file_path = fullfile( obj.img_dir , [ int2str(M.image_id) '.jpg' ] );
        end
        
        function M = meta(obj)
            M = obj.img_data(obj.current());
        end
        function [I, im_path] = image(obj)
            local_file_path = obj.image_path;
            assert( exist(local_file_path,'file')~=0, 'local image does not exist' ); % use url as fallback?
            %I = cached_file_func( @imread, local_file_path, 'images', 1000 );
            I = obj.get_data();
            im_path = local_file_path;
        end
    end
end
