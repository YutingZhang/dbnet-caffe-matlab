classdef nldetRegionProposalNet_Cached < handle
    
    properties (GetAccess = public, SetAccess = protected)
        cache_folder
        max_box_num
        img_ids
        boxes
        
        transpose = 0
        coordinate_base = 1
        coordinate_type
    end
    
    properties ( Constant, GetAccess = public )
        use_caffe_net = 0
        default_param = scalar_struct( ...
            'Maximum_BoxNum', 1000 );
    end
    
    methods
        function obj = nldetRegionProposalNet_Cached( block_def, PARAM )
            
            cache_dir_fn = fullfile(block_def.def_dir,'cache_dir.txt');
            cache_folder_cell = readlines( cache_dir_fn, 1 );
            obj.cache_folder = cache_folder_cell{1};
            
            obj.max_box_num  = PARAM.Maximum_BoxNum;
            
            net_param_fn = fullfile(block_def.def_dir,'param.prototxt');
            net_param = prototxt2struct( net_param_fn, [], 0 );
            obj.transpose       = net_param.transpose;
            obj.coordinate_base = net_param.coordinate_base;
            if isfield( net_param, 'coordinate_type' )
                obj.coordinate_type = net_param.coordinate_type;
            else
                obj.coordinate_type = 'xywh';
            end
        end
        
        function SetInputData( obj, img_ids )
            
            obj.img_ids = img_ids;
            
        end
        
        function Forward( obj )
            
            B = cell( numel(obj.img_ids), 1 );
            for k = 1:numel(obj.img_ids)
                if iscell(obj.img_ids)
                    fn_only = obj.img_ids{k};
                    assert( ischar(fn_only), 'wrong img_ids or img_filename' );
                else
                    fn_only = int2str(obj.img_ids(k));
                end
                fn = fullfile( obj.cache_folder, [fn_only '.mat'] );
                if exist(fn,'file')
                    % B_k = cached_file_func( @load, fn, 'rpn-cache', 100 );
                    B_k = load7_try_single(fn);
                    if isstruct(B_k)
                        B_k_struct = B_k;
                        if obj.transpose
                            B_k = [B_k_struct.boxes.',B_k_struct.box_scores.'];
                        else
                            B_k = [B_k_struct.boxes,B_k_struct.box_scores];
                        end
                    else
                        if obj.transpose
                            B_k = B_k.';
                        end
                    end
                    [~,sidx] = sort(B_k(:,5),'descend');
                    B_k = B_k(sidx,:);
                    the_coordinate_type = default_eval('obj.coordinate_type', 'ltrb'); % fallback implmentation ********
                    switch the_coordinate_type
                        case 'xywh'
                            B_k(:,3:4) = B_k(:,1:2)+B_k(:,3:4)-1; %[x1,y1,w,h] -> [x1,y1,x2,y2]
                            B_k(:,[1,2,3,4]) = B_k(:,[2 1 4 3]);  % [y1,x1,y2,x2] -> [y1,x1,y2,x2]
                        case 'ltrb'
                            B_k(:,[1,2,3,4]) = B_k(:,[2 1 4 3]);  % [y1,x1,y2,x2] -> [y1,x1,y2,x2]
                        case 'tlbr'
                            % nothing
                        otherwise
                            error( 'unrecognized coordinate type' );
                    end
                    B_k = B_k(1:min(end,obj.max_box_num),[1,2,3,4]);
                    B_k = B_k + (1-obj.coordinate_base);
                    B_k(:,5) = k;
                    B{k} = B_k(:,[5 1 2 3 4]); % image_id, y1x1y2x2
                else
                    error( 'Cannot load box cache: %s', fn );
                end
                if isempty(B{k})
                    B{k} = zeros(0,5);
                end
            end
            B = cat(1,B{:});
            % obj.boxes = permute(B, [3, 4, 2, 1]);
            obj.boxes = single(B);
            
        end
        
        function boxes = GetOutputData( obj )
            boxes = obj.boxes;
        end

        function SetOutputDiff( obj, boxes_diff )
            % do nothing
        end
        
        function Backward( obj )
            % do nothing
        end
        
        function Update( obj )
            % do nothing
        end

        
    end
    
end
