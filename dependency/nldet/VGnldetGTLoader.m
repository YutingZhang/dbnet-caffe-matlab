classdef VGnldetGTLoader < handle
    properties ( Access = protected )
        gt_regions
        gt_region_base_source
        gt_text_comp
        gt_cell
        gt_phrase_lookup_table
        
        param
    end
    methods
        function obj = VGnldetGTLoader( DataSubset_Name, ...
                SpecificDir4PrepDataset, PARAM )
            if nargin<3
                PARAM = struct();
            end
            Pdef = struct();
            Pdef.ann_name_regions             = [];
            Pdef.ann_name_similarity_scores   = [];
            Pdef.ann_name_phrase_lookup_table = [];
            obj.param = xmerge_struct(Pdef,PARAM);
            assert( ~isempty( obj.param.ann_name_regions ), 'must set up ann_name_regions' );
            
        end
        
        % read raw data
        function S = gtRegions(obj)
            if isempty(obj.gt_regions)
                region_names = obj.param.ann_name_regions;
                if ~iscell(obj.param.ann_name_regions)
                    region_names = {region_names};
                end
                obj_gt_regions = cell( numel(region_names), 1 );
                obj_gt_region_base_source = zeros( numel(region_names), 1 );
                for k = 1:numel(region_names)
                    obj_gt_regions{k} = vec(VGAnnotation( region_names{k} )).';
                    if k>1
                        assert(numel(obj_gt_regions{1})==numel(obj_gt_regions{k}), ...
                            'multiple region annotations should have consistent number of images' );
                        try
                            RGN_km1 = cat(2, obj_gt_regions{k-1}.regions);
                        catch
                            RGN_km1 = cat(1, obj_gt_regions{k-1}.regions);
                        end
                        if isfield( RGN_km1, 'categ_id' )
                            pre_max_source = max(cat(1,RGN_km1.categ_id));
                        else
                            pre_max_source = max(cat(1,RGN_km1.categ_id));
                        end
                        obj_gt_region_base_source(k) = ...
                            obj_gt_region_base_source(k-1) + pre_max_source;
                        clear RGN_km1
                    end
                end
                obj.gt_regions = obj_gt_regions;
                obj.gt_region_base_source = obj_gt_region_base_source;
            end
            S = obj.gt_regions;
        end

        function c = has_text_comp( obj )
            c = ~isempty( obj.param.ann_name_similarity_scores );
        end
        
        % read text compatibility
        function C = gtTextCompatibility(obj)
            if isempty(obj.gt_text_comp)
                C = VGAnnotation( obj.param.ann_name_similarity_scores );
                C = max(C,C.');
                C(1:(size(C,1)+1):end) = 1;
                obj.gt_text_comp = C;
            end
            C = obj.gt_text_comp;
        end

        function c = has_phrase( obj )
            c = ~isempty( obj.param.ann_name_phrase_lookup_table );
        end
        
        % read phrase vocab
        function P = gtPhrasesVocab(obj)
            if isempty(obj.gt_phrase_lookup_table)
                obj.gt_phrase_lookup_table = VGAnnotation( ...
                    obj.param.ann_name_phrase_lookup_table );
            end
            P = obj.gt_phrase_lookup_table;
        end
        
        % get annotation for 
        function A = get( obj, image_order )
            if isempty(obj.gt_regions)
                A = obj.get_struct_at( image_order );
                obj.gt_cell = cell( numel(obj.gt_regions{1}), 1 );
                obj.gt_cell{image_order} = A;
            end
            A = obj.gt_cell{image_order};
            if isempty(A)
                A = obj.get_struct_at( image_order );
                obj.gt_cell{image_order} = A;
            end
        end
        
        % get text compitability (optional)
        function A = text_comp( obj, query_ids, gallary_ids )
            A = read_text_compatibility( query_ids, gallary_ids, ...
                obj.gtTextCompatibility );
        end
        
        % get phrases (optional, if defined can improve the efficiency for hard mining)
        
        function A = phrase( obj, source_id )
            % assume categ id is available
            S = obj.gtRegions();
            P = obj.gtPhrasesVocab();
            A = cell(numel(source_id),1);
            for k = 1:numel(A)
                irid = P.dict(source_id(k),:);
                A{k} = S(irid(1)).regions(irid(2)).phrase;
            end
        end
        
    end
    
    methods (Access = protected)
        % get the annotations for a particular image
        function A = get_struct_at( obj, image_order )
            S = obj.gtRegions();
            A = cell(numel(S),1);
            for k = 1:numel(S)
                A{k} = obj.get_struct_at_component( image_order, S, k );
            end
            A = cat(1,A{:});
        end
        function A = get_struct_at_component( obj, image_order, S, component_id )
            R = S{component_id}(image_order);
            R = R.regions;
            
            if isempty(R)
                A = struct( 'box', {}, 'text', {} );
                return;
            end
            
            BOXES = [[R.y];[R.x];[R.height];[R.width]].';
            BOXES = [BOXES(:,1:2),(BOXES(:,1:2)+BOXES(:,3:4)-1)]+1; % tlbr: [y1,x1,y2,x2], 0-base to 1-base
            BOXES = single(BOXES);
            PRHASES = {R.phrase}.';
            if isfield(R,'categ_id')
                source_arr = [R.categ_id].';
            else
                source_arr = [R.region_id].';
            end
            source_arr = source_arr + obj.gt_region_base_source(component_id);
            source_cell = num2cell( source_arr );
            T = struct('phrase',PRHASES,'source',source_cell);
            A = struct( 'box', mat2cell(BOXES,ones(size(BOXES,1),1),4), ...
                'text', num2cell(T) );
        end
    end
end
