classdef VOCnldetGTLoader < handle
    properties ( GetAccess = public, SetAccess = protected )
        img_data
        
        subsetNames
        SpecificDir4PrepDataset
        
        use_difficult
    end
    methods
        function obj = VOCnldetGTLoader( subsetNames, SpecificDir4PrepDataset, use_difficult )
            if ~exist('SpecificDir4PrepDataset','var')
                SpecificDir4PrepDataset = evalin( 'caller', 'SPECIFIC_DIRS.PrepDataset' );
            end
            obj.subsetNames = subsetNames;
            obj.SpecificDir4PrepDataset = SpecificDir4PrepDataset;
            if nargin<3 || isempty(use_difficult)
                if ismember(subsetNames,{'trainval','train'})
                    use_difficult = 1;
                else
                    use_difficult = 0;
                end
            end
            obj.use_difficult = use_difficult;
        end
        
        % read raw data
        function S = gtRegions(obj)
            if isempty(obj.img_data)
                CATEG_LIST = VOCCategories;
                image_data = VOCImageSampler.load_image_data( ...
                    obj.subsetNames, obj.SpecificDir4PrepDataset );
                t1 = tic_print( '[reformat VOC annotation] ' );
                for k = 1:numel(image_data)
                    if obj.use_difficult
                        image_data(k).obj( [image_data(k).obj.difficulty]>0 ) = [];
                    end
                    [~,type_id] = ismember( {image_data(k).obj.type}, CATEG_LIST );
                    type_id = num2cell(type_id);
                    [image_data(k).obj.type_id] = deal(type_id{:});
                end
                obj.img_data = image_data;
                toc_print(t1);
            end
            S = obj.img_data;
        end
        
        % get the annotations for a particular image
        function A = get( obj, image_order )
            S = obj.gtRegions();
            R = S(image_order);
            R = R.obj;
            A = repmat( struct(), numel(R), 1 );
            for k = 1:numel(R)
                A(k).box  = single([R(k).y1, R(k).x1, R(k).y2, R(k).x2]);
                A(k).text = struct( 'phrase', {R(k).type}, 'source', {R(k).type_id} );
            end
        end
    end
end
