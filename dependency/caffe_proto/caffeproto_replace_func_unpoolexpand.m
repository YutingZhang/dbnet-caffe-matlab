function [ X, varargout ] = caffeproto_replace_func_unpoolexpand( varargin )

if strcmp(varargin{1},'extend')
    X = {  
        @(varargin) caffeproto_replace_func_unpoolexpand( 'normalization', varargin{:} ), ...
        @(varargin) caffeproto_replace_func_unpoolexpand( 'pool-type-expand', varargin{:} ), ...
        @(varargin) caffeproto_replace_func_unpoolexpand( 'normalization', varargin{:} )
        };
    return;
end

VAR_IN = varargin(2:end);

if ~isempty(VAR_IN) && strcmp( VAR_IN{1}, 'extend' )
    X = {};
    return;
elseif strcmp(VAR_IN{1},'adjacent')
    X = [0];
    return;
end

subS = VAR_IN{1};
if ~strcmp( subS.type{1}, 'Depooling' )
    X = [];
    return;
end

switch varargin{1}
    case 'normalization'   
        X = subS;
        if isfield( X, 'pooling_param' ) && isfield( X.pooling_param, 'normalized' )
            isNormalizedOutput = X.pooling_param.normalized;
            X.pooling_param = rmfield(X.pooling_param,'normalized');
            if isNormalizedOutput
                finalTop  = X(1).top{1};
                layerName = X(1).name{1}; 
                if length(X(1).top) <= 1
                    X(1).top    = { [finalTop '/unnorm'], [finalTop '/mask'] };
                else
                    X(1).top{1} = [finalTop '/unnorm'];
                end
                X(2).name   = { [ layerName '/mask-inv' ] };
                X(2).type   = { 'SafeInv' };
                X(2).bottom = X(1).top(2);
                X(2).top    = { [finalTop '/mask-inv'] };
                X(3).name   = { [ layerName '/norm' ] };
                X(3).type   = { 'Eltwise' };
                X(3).bottom = [ X(1).top(1), X(2).top(1) ];
                X(3).top{1} = finalTop;
                X(3).eltwise_param.operation = pbEnum('PROD');
            end
        end
    case 'pool-type-expand'
        X = subS;
        if isfield( subS, 'pooling_param' ) && isfield( subS.pooling_param, 'pool' ) ...
                && ( length(subS.pooling_param.pool)>=2 || isfield( subS.pooling_param, 'bias_weight' ) )
            weight_learnable = 0; learnable_tag = '';
            if isfield( X.pooling_param, 'weight_learnable' )
                if X.pooling_param.weight_learnable
                    weight_learnable = 1; learnable_tag = '*';
                end
                X.pooling_param  = rmfield(X.pooling_param,'weight_learnable');
            end
            weight_lr_mult = 1;
            if isfield( X.pooling_param, 'weight_lr_mult' )
                weight_lr_mult = X.pooling_param.weight_lr_mult;
                X.pooling_param  = rmfield(X.pooling_param,'weight_lr_mult');
            end
            numSubPool  = length(X.pooling_param.pool);
            mult_weights = ones(1,numSubPool);
            if isfield( X.pooling_param, 'mult_weights' )
                mult_weights = X.pooling_param.mult_weights;
                X.pooling_param = rmfield(X.pooling_param,'mult_weights');
            end
            if length(X.top) < 2
                mult_weights = [];
            end
            mult_scales = ones(1,numSubPool);
            if isfield( X.pooling_param, 'mult_scales' )
                mult_scales = X.pooling_param.mult_scales;
                X.pooling_param = rmfield(X.pooling_param,'mult_scales');
            end
            bias_weight = 0;
            if isfield( X.pooling_param, 'bias_weight' )
                bias_weight = X.pooling_param.bias_weight;
                X.pooling_param = rmfield(X.pooling_param,'bias_weight');
            end
            array_normalized = false(1,numSubPool);
            if isfield( X.pooling_param, 'array_normalized' )
                array_normalized = X.pooling_param.array_normalized;
                X.pooling_param = rmfield(X.pooling_param,'array_normalized');
            end
            
            poolTypes  = X.pooling_param.pool;
            X0 = X;
            X  = repmat(X0,1,numSubPool);
            Xmerge_bottom = cell(1,numSubPool);
            for k=1:numSubPool
                X(k).name   = { [X0.name{1} '!' poolTypes(k).val()] };
                X(k).bottom{1} = X0.bottom{1};
                if length(X0.bottom)>k && ~isempty(X0.bottom{k+1})
                    X(k).bottom{2} = X0.bottom{k+1};
                end
                X(k).top{1} = [X0.top{1} '!' poolTypes(k).val()];
                Xmerge_bottom{k} = X(k).top{1} ;
                X(k).pooling_param.pool = poolTypes(k);
                X(k).pooling_param.normalized = array_normalized(k);
            end
            Xmerge = struct();
            Xmerge.name   = { [X0.name{1} '/merge' learnable_tag] };
            Xmerge.type   = {'Eltwise'};
            Xmerge.bottom = Xmerge_bottom;
            Xmerge.top    = X0.top(1);
            Xmerge.eltwise_param.operation = pbEnum('SUM');
            
            if isempty( mult_weights )
                Xmerge.eltwise_param.coeff = mult_scales;
                Xmask_merge = struct([]);
            else
                Xmerge.eltwise_param.coeff = mult_scales.*mult_weights;
                Xmask_merge_bottom = cell(1,numSubPool);
                for k=1:numSubPool
                    X(k).top{2} = [X0.top{2} '!' poolTypes(k).val()];
                    Xmask_merge_bottom{k} = X(k).top{2};
                end
                Xmask_merge = struct();
                Xmask_merge.name   = { [X0.name{1} '/mask/merge' learnable_tag] };
                Xmask_merge.type   = {'Eltwise'};
                Xmask_merge.bottom = Xmask_merge_bottom;
                Xmask_merge.top    = { [X0.top{2} '-pre'] };
                Xmask_merge.eltwise_param.operation = pbEnum('SUM');
                Xmask_merge.eltwise_param.coeff  = mult_weights;
                
                Xmask_merge(2).name   = { [X0.name{1} '/mask/add-bias' learnable_tag] };
                Xmask_merge(2).type   = {'Power'};
                Xmask_merge(2).bottom = Xmask_merge(1).top;
                Xmask_merge(2).top    = X0.top(2);
                Xmask_merge(2).power_param.shift = bias_weight;
                if weight_learnable
                    if length(subS.pooling_param.pool)>1
                        Xmask_merge(1).eltwise_param.learnable_coeff = true;
                        Xmask_merge(1).param.lr_mult = weight_lr_mult;
                        Xmask_merge(1).param.decay_mult = 1;
                    end
                    if bias_weight
                        Xmask_merge(2).power_param.learnable_shift = true;
                        Xmask_merge(2).power_param.nonnegative_shift = true;
                        %Xmask_merge(2).param = struct;
                        Xmask_merge(2).param = struct('lr_mult',0,'decay_mult',1);
                        Xmask_merge(2).param(2) = struct('lr_mult',0,'decay_mult',1);
                        Xmask_merge(2).param(3) = struct('lr_mult',weight_lr_mult,'decay_mult',1);
                    end
                end
            end
            if weight_learnable
                if length(subS.pooling_param.pool)>1
                    Xmerge.eltwise_param.learnable_coeff = true;
                    Xmerge.param.lr_mult = 1;
                    Xmerge.param.decay_mult = 1;
                end
            end
            X  = cat_struct(2,X,Xmerge,Xmask_merge);
            XA = struct( 'layer', {X} );
            XA = caffeproto_replace(XA,'simplifyeltwise');
            X  = XA.layer;
        end
    otherwise
        error( 'unrecognized branch' );
end

