classdef VOCImageSampler < IndexSampler
    properties ( GetAccess = public, SetAccess = protected )
        img_data
    end
    methods
        function obj = VOCImageSampler( subsetNames, PARAM, SpecificDir4PrepDataset )
            % VOCImageSampler( subsetImageIndexes, PARAM [, SpecificDir4PrepDataset] )
            % subsetImageIndexes: 'trainval', 'train', 'val', 'test', 
            %                      logical or numerical indexes. 
            %                      or cell array of any them
            if nargin<2, PARAM = struct(); end
            if ~exist('SpecificDir4PrepDataset','var')
                SpecificDir4PrepDataset = evalin( 'caller', 'SPECIFIC_DIRS.PrepDataset' );
            end
            IMAGE_DATA = VOCImageSampler.load_image_data( ...
                subsetNames,SpecificDir4PrepDataset );
            
            chosenImageIndexes = true( numel(IMAGE_DATA), 1 );
            
            obj = obj@IndexSampler(chosenImageIndexes, PARAM);
            obj.img_data = IMAGE_DATA;
            
        end
        function M = meta(obj)
            R = obj.img_data(obj.current());
            [~,fn,~] = fileparts( R.im );
            M = struct( 'image_id', {fn}, 'im_path', {R.im}, ...
                'is_annotated', {R.is_annotated} );
        end
        function [I, im_path] = image(obj)
            R = obj.img_data(obj.current());
            local_file_path = R.im;
            assert( exist(local_file_path,'file')~=0, 'local image does not exist' );
            I = cached_file_func( @imread, local_file_path, 'images', 1000 );
            im_path = local_file_path;
        end
    end
    
    methods (Static, Access=public)
        function IMAGE_DATA = load_image_data( subsetNames, SpecificDir4PrepDataset )
            if ~exist('SpecificDir4PrepDataset','var')
                SpecificDir4PrepDataset = evalin( 'caller', 'SPECIFIC_DIRS.PrepDataset' );
            end

            if ~iscell(subsetNames)
                subsetNames = {subsetNames};
            end
            
            SUBSETS = prototxt2struct( fullfile(SpecificDir4PrepDataset, 'meta.prototxt') );
            chosen_subsets = {};
            for k = 1:numel(subsetNames)
                curSubsetName = subsetNames{k};
                assert( ischar(curSubsetName), 'subsetName must be char' );
                if ~isfield( SUBSETS, curSubsetName )
                    error('Unrecognized subsetImageIndexes');
                end
                cur_chosen_subsets = subsref( SUBSETS, substruct('.',curSubsetName) );
                chosen_subsets = [ chosen_subsets, cur_chosen_subsets];
            end
            chosen_subsets = unique(chosen_subsets,'stable');
            IMAGE_DATA = cell(numel(chosen_subsets),1);
            for k = 1:numel(chosen_subsets)
                IMAGE_DATA{k} = vec( VOCAnnotation( chosen_subsets{k} ) );
            end
            IMAGE_DATA = cat_struct(1,IMAGE_DATA{:});
        end
    end
end
