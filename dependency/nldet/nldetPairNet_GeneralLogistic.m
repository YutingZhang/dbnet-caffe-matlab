classdef nldetPairNet_GeneralLogistic < nldetCaffeBlock 
        
    properties ( Constant, GetAccess = public )
        use_caffe_net = 1
        default_param = scalar_struct( ...
            'PositiveWeight',     1, ...
            'LocalizationWeight', 1, ... 
            'Positive_RelaxedTopNum',     0, ...
            'Positive_ChosenTopNum',      Inf, ...
            'Localization_RelaxedTopNum', 0, ...
            'Localization_ChosenTopNum',  Inf, ...
            'Negative_RelaxedTopNum',     0, ...
            'Negative_ChosenTopNum',      Inf );
    end

    properties ( GetAccess = public, SetAccess = protected )
        param
        label
        loss
        
        prob
        
        loss_weight
        
        
        pos_weight
        loc_weight
        neg_weight
        
        diff_weight
        
        is_train
        
    end
    
    methods ( Static )
        function cleaned_idxb = choose_violated( ...
                raw_idxb, loss, top_num, relaxed_num )
            num_raw = sum(raw_idxb);
            if num_raw==0,
                cleaned_idxb = raw_idxb;
                return;
            end
            assert( top_num>0, 'top_num must be greater than 0' );
            assert( relaxed_num>=0, 'relaxed_num must be no less than 0' );
            if top_num<1 % it can be a percentage input
                top_num = ceil( num_raw*top_num );
            end
            if relaxed_num<1 % it can be a percentage input
                relaxed_num = floor( num_raw*relaxed_num );
            end
            if relaxed_num == 0 && sum(num_raw)<=top_num
                cleaned_idxb = raw_idxb;
                return;
            end
            top_num = min( top_num, num_raw-relaxed_num );
            raw_idx = find(raw_idxb);
            [~,sidx] = sort( loss(raw_idxb), 'descend' );
            cleaned_idx = raw_idx( sidx( (relaxed_num+1):(relaxed_num+top_num) ) );
            cleaned_idxb = false(size(raw_idxb));
            cleaned_idxb(cleaned_idx) = true;
        end
    end
    
    methods
        
        function obj = nldetPairNet_GeneralLogistic( block, PARAM )
            
            obj@nldetCaffeBlock(block);
            
            obj.param = PARAM;
            
            total_weight = obj.param.PositiveWeight + ...
                obj.param.LocalizationWeight + 1;
            
            obj.pos_weight = obj.param.PositiveWeight / total_weight;
            obj.loc_weight = obj.param.LocalizationWeight / total_weight;
            obj.neg_weight = 1 - ( obj.pos_weight + obj.loc_weight );
            
            obj.is_train = block.aux.block_def.is_train;
            
        end
        
        function SetInputData( obj, im_feat, dy_param, label )
            
            obj.SetInputData_( im_feat, dy_param );
            if ~obj.is_train, return; end
            
            obj.label = label;
            
        end
        
        function Forward( obj )
            
            obj.Forward_();
            
            z = obj.GetOutputData_();
            if isa( z, 'gpuArray' )
                if ~isa( obj.label, 'gpuArray' )
                    obj.label = gpuArray( single(obj.label) );
                end
            end            
            
            p = zeros( size(z), 'like', z);
            z_pos_idxb = (z>0);
            p(z_pos_idxb) = 1./(1+exp(-z(z_pos_idxb)));
            exp_z_neg = exp(z(~z_pos_idxb));
            p(~z_pos_idxb)  = exp_z_neg./(1+exp_z_neg);
            obj.prob  = p;
            
            if ~obj.is_train, return; end
            
            y = reshape(obj.label, size(z) );
            
            l = z-y.*z-log(obj.prob);

            pos_idxb = (y(:) == 1);
            loc_idxb = (y(:)>0 & y(:)<1); 
            neg_idxb = (y(:) == 0);
            
            pos_idxb = obj.choose_violated( pos_idxb, l, ...
                obj.param.Positive_ChosenTopNum, ...
                obj.param.Positive_RelaxedTopNum );
            loc_idxb = obj.choose_violated( loc_idxb, l, ...
                obj.param.Localization_ChosenTopNum, ...
                obj.param.Localization_RelaxedTopNum );
            neg_idxb = obj.choose_violated( neg_idxb, l, ...
                obj.param.Negative_ChosenTopNum, ...
                obj.param.Negative_RelaxedTopNum );
            
            loss_pos = mean(l(pos_idxb));
            loss_loc = mean(l(loc_idxb));
            loss_neg = mean(l(neg_idxb));
            obj.loss = gather( [loss_pos,loss_neg,loss_loc] );
            
            obj.diff_weight = zeros( size(z), 'like', z );
            obj.diff_weight(pos_idxb) = obj.pos_weight ./ sum(pos_idxb);
            obj.diff_weight(loc_idxb) = obj.loc_weight ./ sum(loc_idxb);
            obj.diff_weight(neg_idxb) = obj.neg_weight ./ sum(neg_idxb);
            
        end
        
        function Backward( obj )
            
            if ~obj.is_train, return; end
            
            y = reshape(obj.label,size(obj.prob));
            z_diff = ( obj.prob-y ) .* obj.diff_weight;
            
            obj.SetOutputDiff_( obj.block.output_names{1}, z_diff );
            obj.Backward_();
            
        end
        
        function ForwardBackward( obj )
            
            obj.Forward();
            obj.Backward();
            
        end

        function varargout = GetOutputData( obj )
            % train: [loss_pos, [loss_neg, loss_loc]]
            % test: [prob]
            if obj.is_train
                if isnan(obj.loss(3))
                    neg_loss = obj.loss(2);
                else
                    neg_loss = obj.loss(2:3);
                end
                varargout = {obj.loss(1), neg_loss, obj.prob};
            else
                varargout = {obj.prob};
            end
        end
        
        function SetOutputDiff( obj, varargin )
            error( 'No output diff can be set.' );
        end

        
    end
    
end
