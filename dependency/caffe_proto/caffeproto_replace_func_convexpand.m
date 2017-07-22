function X = caffeproto_replace_func_convexpand( varargin )


if ischar(varargin{1})
    if strcmp(varargin{1},'extend')
        X = { ... 
            @(varargin) caffeproto_replace_func_convexpand( 'unit-expand', varargin{:} ), ...
            @(varargin) caffeproto_replace_func_convexpand( 'convcomb', varargin{:} ), ...
            [ {@(varargin) caffeproto_replace_func_remove_split( varargin{:} ) }, {'extra_condition', ...
            @(subS) default_eval('subS.origin_from_convexpand',false) } ]
            };
        return;
    end
end

VAR_IN = varargin(2:end);
if ~isempty(VAR_IN) && strcmp( VAR_IN{1}, 'extend' )
    X = {};
    return;
end

switch varargin{1}
    case 'unit-expand'
        % unit expand;
        if isequal( VAR_IN{1}, 'adjacent' )
            X = [0];
            return;
        end

        S = VAR_IN{1};
        X = [];
        if ~strcmp(S.type{1},'ConvCombo'), return; end
        
        % handle side res
        srP = default_eval( 'S.conv_combo_param.side_res', [] );
        X0 = S;
        X0.conv_combo_param.side_res = [];
        srX = struct([]);
        if ~isempty( srP )
            srP0 = struct();
            srP0.pre_shape = []; % use pooling to reshape
            srP0.pre_layers = [];
            srP0.mid_shape  = []; % if not specified, then the same as pre_shape
            srP0.base_weight = 1; % factor for previous layer
            srP0.cat_back = 0; % link memory unit back to original stream
            srP0.post_layers = []; % include loss layer here, can be empty
            
            if isstruct( srP )
                srP = {srP};
            end
            srP = cellfun( @(a) xmerge_struct( 'always', 'always', srP0, a ), ...
                srP, 'UniformOutput', 0 );
            
            side_prefix_func = @(k) sprintf( '%s/SR%d/', S.top{1}, k);
            side_bottom_prefix_func = @(k) sprintf( '%s/SR%d/', S.bottom{1}, k);
            top_shape = caffeproto_avail_ioshape( S, 't', 1 );
            top_shape = top_shape(2:end);
            curS = 'X0';
            for k = 1:numel(srP)
                if isempty( srP{k}.pre_shape )
                    assert( ~isempty(top_shape), 'cannot figure out the blob shape' );
                    pre_shape = top_shape;
                else
                    pre_shape = srP{k}.pre_shape;
                    if isscalar(pre_shape) && ~isempty(top_shape)
                        pre_shape = reshape( pre_shape, size(top_shape) );
                    end
                end
                
                top_name = eval( [curS '.top{1}'] );
                side_prefix = side_prefix_func(k);
                
                if isempty( srP{k}.mid_shape )
                    mid_shape = pre_shape;
                else
                    mid_shape = srP{k}.mid_shape;
                    if isscalar(mid_shape)
                        mid_shape = reshape( mid_shape, pre_shape );
                    end
                end
                % add pre pooling layer as needed
                curS1 = curS;
                X = struct([]);
                curTop = top_name;
                if ~isempty(top_shape)
                    pre_reshape_factor = top_shape./pre_shape;
                    assert( all(round(pre_reshape_factor)==pre_reshape_factor), ...
                        'non integer factor' );
                    if any(pre_reshape_factor>1)
                        G = struct();
                        G.name = { [side_prefix '*pool'] };
                        G.type = { 'Pooling' };
                        G.bottom = {curTop};
                        G.top = G.name;
                        G.pooling_param.pool = pbEnum( 'AVE' );
                        assert(numel(pre_reshape_factor)==2, 'pool only support 2d');
                        if pre_reshape_factor(1) == pre_reshape_factor(2)
                            G.pooling_param.kernel_size = pre_reshape_factor(1);
                            G.pooling_param.stride = G.pooling_param.kernel_size;
                        else
                            G.pooling_param.kernel_h = pre_reshape_factor(1);
                            G.pooling_param.kernel_w = pre_reshape_factor(2);
                            G.pooling_param.stride_h = G.pooling_param.kernel_h;
                            G.pooling_param.stride_w = G.pooling_param.kernel_w;
                        end
                        X = cat_struct(2,X,G);
                        curTop = G.top{1};
                        curS1 = sprintf( 'X(%d)', numel(X) );
                    end
                end
                % prefix layers
                pre_layers = srP{k}.pre_layers;
                post_layers = srP{k}.post_layers;
                if isempty(post_layers)
                    if ~isempty(pre_layers)
                        pre_layers = caffeproto_prefix_net( ...
                            struct( 'layer', {pre_layers} ), [] , side_prefix );
                        pre_layers = pre_layers.layer;
                    end
                else
                    if isempty(pre_layers)
                        post_layers = caffeproto_prefix_net( ...
                            struct( 'layer', {post_layers} ), [] , side_prefix );
                        post_layers = post_layers.layer;
                    else
                        all_layers = cat_struct(2,pre_layers,post_layers);
                        all_layers = caffeproto_prefix_net( ...
                            struct( 'layer', {all_layers} ), [] , side_prefix );
                        all_layers = all_layers.layer;
                        pre_layers  = all_layers(1:numel(pre_layers));
                        post_layers = all_layers((numel(pre_layers)+1):end);
                    end
                end
                
                % add prelayers
                                
                if ~isempty(pre_layers)
                    pre_layers(1).bottom{1} = curTop;
                    X = cat_struct(2,X,pre_layers);
                    curTop = pre_layers(end).top{1};
                end
                % add res
                base_weight = srP{k}.base_weight;
                if ~isempty(pre_layers)
                    side_bottom_prefix = side_bottom_prefix_func(k);
                    G = struct();
                    G.name = { [side_prefix '*res'] };
                    if base_weight
                        G.type = { 'Eltwise' };
                        G.bottom = { [side_bottom_prefix '*res'], curTop };
                        G.eltwise_param.operation = pbEnum('SUM');
                        G.eltwise_param.coeff = [ base_weight, 1 ];
                    else
                        G.type = { 'Split:Branch' };
                        G.bottom = { curTop };
                    end
                    G.top = G.name;
                    X = cat_struct(2,X,G);
                    curTop = G.top{1};
                    curS1 = sprintf( 'X(%d)', numel(X) );
                end
                % add cat back
                cat_back = srP{k}.cat_back;
                if cat_back
                    assert( k==1, 'do not support cat_back for multi layer res' );
                    cat_bak_rescale_factor = top_shape./mid_shape;
                    assert( all( floor(cat_bak_rescale_factor) == cat_bak_rescale_factor ), ...
                        'need interger factor' );
                    catbackTop = curTop;
                    if any(cat_bak_rescale_factor>1)
                        catbackTop = [curTop '/*unpool'];
                        G = struct();
                        G.name = { catbackTop };
                        G.type = { 'Depooling' };
                        G.bottom = {curTop};
                        G.top  = G.name;
                        G.pooling_param.pool = pbEnum( 'SUM' );
                        assert(numel(cat_bak_rescale_factor)==2, 'pool only support 2d');
                        if cat_bak_rescale_factor(1) == cat_bak_rescale_factor(2)
                            G.pooling_param.kernel_size = cat_bak_rescale_factor(1);
                            G.pooling_param.stride = G.pooling_param.kernel_size;
                        else
                            G.pooling_param.kernel_h = cat_bak_rescale_factor(1);
                            G.pooling_param.kernel_w = cat_bak_rescale_factor(2);
                            G.pooling_param.stride_h = G.pooling_param.kernel_h;
                            G.pooling_param.stride_w = G.pooling_param.kernel_w;
                        end
                        X = cat_struct(2,X,G);
                    end
                    G = struct();
                    G.name = { [top_name '/cat_back'] };
                    G.type = { 'Concat' };
                    G.bottom = { top_name, catbackTop };
                    G.top  = { top_name };
                    G.TOP  = { [top_name '/cat_back'] };
                    X = cat_struct(2,X,G);
                end
                % add post layers
                if ~isempty(post_layers)
                    post_layers(1).bottom{1} = curTop;
                    X = cat_struct(2,X,post_layers);
                end
                curS = curS1;
            end
            srX = X;
        end
        X = struct([]);
        X = cat_struct(2,X,srX);
        if isempty(X)
            X = [];
        else
            X = cat_struct(2,X0,X);
        end
        
    case 'convcomb'
        if isequal( VAR_IN{1}, 'adjacent' )
            X = [0];
            return;
        end

        S = VAR_IN{1};

        if ~strcmp(S.type{1},'ConvCombo')
            X = [];
            return;
        end

        P = S.conv_combo_param;

        X = partial_struct(S,'@exclude','conv_combo_param', ...
            'pooling_param', 'lrn_param', 'dropout_param', 'reshape_param', ...
            'relu_param', 'bn_param', ...
            'noise_param_before', 'noise_param_after', ...
            'propagate_down', 'aux' );
        [auxB, auxT] = caffeproto_convcombo_get_aux_blobs( S, 'ConvCombo' );
        X.bottom = [S.bottom, auxB];
        X.top    = [S.top, auxT];
        X_side = struct([]);
        
        if ismember( P.type, {'Pooling','Depooling'} )
            X.pooling_param = S.pooling_param;
        end
        
        G0 = empty_struct(X);
        for k = 1:length(P.aux_layers)
            G = G0;
            posA = 'after';
            subPostfix = [];
            subInplace = 0;
            layerTypeWithTag = P.aux_layers{k};
            layerTagSepIdx = find(layerTypeWithTag==':',1);
            if isempty(layerTagSepIdx)
                layerType = layerTypeWithTag;
                layerTag  = [];
            else
                layerType = layerTypeWithTag(1:layerTagSepIdx-1);
                layerTag  = layerTypeWithTag(layerTagSepIdx+1:end);
            end
            switch layerType
                case 'ReLU'
                    G.type{1}  = 'ReLU';
                    subPostfix = 'relu';
                    is_crelu = default_eval( 'S.relu_param.crelu', false );
                    [G,S] = transfer_field( ...
                        G,S,'relu_param', 0 );
                    subInplace = 1;
                    if is_crelu, subInplace = 0; end
                    posA = 'after';
                case 'Pooling'
                    G.type{1}  = 'Pooling';
                    subPostfix = 'pool';
                    [G,S] = transfer_field( ...
                        G,S,'pooling_param', 0 );
                    if isfield(S.conv_combo_param,'pooling_backprop_switch') && ...
                            ~isempty(S.conv_combo_param.unpooling_backprop_switch)
                        G.propagate_down = [true ...
                            logical(S.conv_combo_param.unpooling_backprop_switch)];
                    end
                    posA = 'after';
                case 'Unpooling'
                    G.type{1}  = 'Depooling';
                    subPostfix = 'unpool';
                    [G,S] = transfer_field( ...
                        G,S,'pooling_param', 0 );
                    if isfield(S.conv_combo_param,'unpooling_backprop_switch')
                        G.propagate_down = [true ...
                            logical(S.conv_combo_param.unpooling_backprop_switch)];
                    end
                    posA = 'before';
                case 'LRN'
                    G.type{1}  = 'LRN';
                    subPostfix = 'lrn';
                    [G,S] = transfer_field( ...
                        G,S,'lrn_param', 0 );
                    posA = 'after';
                case 'BN-side'
                    G.type{1}  = 'BN';
                    subPostfix = 'bn';
                    G.bn_param.bn_mode = pbEnum('AVERAGE');
                    posA = 'side';
                case 'BN'
                    G.type{1}  = 'BN';
                    subPostfix = 'bn';
                    [G,S] = transfer_field( ...
                        G,S,'bn_param', 0 );
                    subInplace1 = try_to_eval( 'G.bn_param.inplace', [] );
                    if ~isempty(subInplace1)
                        G.bn_param = rmfield(G.bn_param,'inplace');
                        if subInplace1
                            subInplace = 1;
                        end
                    end
                    if isfield( G.bn_param, 'param' )
                        G.param    = G.bn_param.param;
                        G.bn_param = rmfield( G.bn_param, 'param' );
                    end
                    if default_eval( 'S.param(1).lr_mult', 1 )
                        if isempty( default_eval( 'G.param', [] ) )
                            G.param = struct( 'lr_mult', {1,1}, 'decay_mult', {1,0} );
                        end
                    else
                        G.param = struct( 'lr_mult', {0,0}, 'decay_mult', {0,0} );
                    end
                    if default_eval( 'S.param(1).lr_mult_zero', 0 )
                        [ G.param.lr_mult ] = rdeal(0);
                    end
                    posA = 'after';
                case 'Reshape'
                    G.type{1}  = 'Reshape';
                    subPostfix = 'reshape';
                    [G,S] = transfer_field( ...
                        G,S,'reshape_param', 0 );
                    posA = 'after';
                case 'Dropout'
                    G.type{1}  = 'Dropout';
                    subPostfix = 'dropout';
                    [G,S] = transfer_field( ...
                        G,S,'dropout_param', 0 );
                    posA = 'before';
                case 'LadderSkip'
                    G.type{1}  = 'Split:Branch';
                    G.tag  = { 'LadderSkip' };
                    subPostfix = 'ladder-skip-0';
                    subInplace = 1;
                    posA = 'after'; 
                case 'LadderCombined'
                    G.type{1}  = 'LadderCombinator';
                    subPostfix = 'ladder-combined';
                    posA = 'after'; 
                case 'AddNoise.Before'
                    G.type{1}  = 'AddNoise';
                    subPostfix = 'add-noise';
                    G.noise_param = try_to_eval( 'S.noise_param_before', [] );
                    posA = 'before';
                case 'AddNoise.After'
                    G.type{1}  = 'AddNoise';
                    subPostfix = 'add-noise';
                    G.noise_param = try_to_eval( 'S.noise_param_after', [] );
                    posA = 'after';
                otherwise
                    error( 'Unrecognized aux layer type' );
            end
            if ~isempty(layerTag)
                subPostfix = [ subPostfix, ':', layerTag ];
            end
            if ismember(posA,{'after','side'})
                G.name{1} = [S.top{1} '/' subPostfix];
                G.bottom = X(end).top(1);
                if subInplace
                    G.top    = G.bottom;
                else
                    G.top{1} = [S.top{1} '/' subPostfix];
                end
            else
                G.name{1} = [S.bottom{1} '/' subPostfix];
                G.bottom  = X(1).bottom(1);
                if subInplace
                    G.top    = G.bottom;
                else
                    G.top{1}  = [G.bottom{1} '/' subPostfix];
                end
                X(1).bottom(1) = G.top(1);
            end
            [auxB, auxT] = caffeproto_convcombo_get_aux_blobs( ...
                S, P.aux_layers{k} );
            auxName = caffeproto_convcombo_get_aux_name( S,P.aux_layers{k} );
            if ischar(auxName), G.name{1} = auxName; end
            G.bottom = [G.bottom auxB];
            if strcmp(posA,'side')
                G.top = [];
                X_side = cat_struct(2,X_side,G);
            else
                G.top = [G.top auxT];
                if strcmp(posA,'after')
                    X = cat_struct(2,X,G);
                else
                    X = cat_struct(2,G,X);
                end
            end
        end
        [~,convLayerId] = ismember('ConvCombo',[X.type]);
        X(convLayerId).type = P.type;

        Xsl = struct([]);
        if isfield( S.conv_combo_param, 'sideloss' )
            if numel(S.conv_combo_param.sideloss)>=1
                assert( numel(S.conv_combo_param.sideloss)==1, ...
                    'Currently only support a single sideloss' );
                slInfo   = S.conv_combo_param.sideloss{1};
                slType   = slInfo{1};
                slPARAM  = slInfo{2};
                top_shape = caffeproto_avail_ioshape( S, 't', 1 );
                top_size = min(top_shape(2:end));
                avePoolSize = 1;
                if slPARAM.maxSideLossEdge<top_size
                     avePoolSize = floor( ...
                         log(top_size/slPARAM.maxSideLossEdge)/log(2) + eps );
                     avePoolSize = 2^(avePoolSize);
                end
                if slPARAM.beforeAux
                    curBottom = S.top{1};
                else
                    curBottom = X(end).top{1};
                end
                Ga = struct([]);
                if avePoolSize>1
                    G  = struct();
                    G.name = { [ S.name{1} '/pre-cls-pool' ] };
                    G.type = { 'Pooling' };
                    G.bottom = { curBottom };
                    G.top    = G.name;
                    G.pooling_param.pool = pbEnum( 'AVE' );
                    G.pooling_param.kernel_size = avePoolSize;
                    G.pooling_param.stride = avePoolSize;
                    Ga = G;
                    curBottom = G.top{1};
                end
                G = struct();
                G.name = { [ S.name{1} '/fc-cls' ] };
                G.type = { 'InnerProduct' };
                G.bottom = { curBottom };
                G.top  = G.name;
                G.inner_product_param = struct();
                G.inner_product_param.num_output = slPARAM.numClasses;
                G.inner_product_param.weight_filler = ...
                    struct('type', {{'gaussian'}}, ...
                    'std', {slPARAM.sidelossGstd} );
                G.inner_product_param.bias_filler = ...
                    struct('type', {{'constant'}}, ...
                    'value', {0} );
                G.param = struct( 'lr_mult', 1, 'decay_mult', {1,0} );
                G1 = G;
                
                G = struct([]);
                switch slType
                    case 'softmax'
                        G = repmat( struct(), 1, 2 ); % 1-SoftmaxWithLoss, 2-Accuracy (TEST)
                        G(1).name = { ['sideloss@' G1.top{1} ] };
                        G(1).type = { 'SoftmaxWithLoss' };
                        G(1).bottom = { G1.top{1}, slPARAM.blobLabel };
                        G(1).top  = G(1).name;
                        G(1).loss_weight = slPARAM.sidelossWeight;
                        G(2).name = { ['accuracy@' G1.top{1} ] };
                        G(2).type = { 'Accuracy' };
                        G(2).bottom = G(1).bottom;
                        G(2).top    = G(2).name;
                        G(2).include = struct( 'phase', { pbEnum('TEST') } );
                    otherwise
                        error( 'invalid sideloss type (internal error)' );
                end
                G2 = G;
                
                Xsl = cat_struct(2,Ga,G1,G2);
            end
        end
        
        G=G0;
        % G.type   = {'ConvCombo-Top'};
        G.name   = {[S.name{1} '/ConvCombo-Top' ]};
        G.type   = {'Split:Branch'};
        G.bottom = X(end).top(1);
        G.top    = S.top;
        G.origin_from_convexpand = true;
        
        X = cat_struct(2,X,X_side,Xsl,G);

%     case 'conv-clean-top'
%         if isequal( VAR_IN{1}, 'adjacent' )
%             X = [0 1; 0 0];
%             return;
%         end
%         
%         subS = VAR_IN{1};
%         is_matched = ismember( subS(1).type, 'ConvCombo-Top' );
%         if is_matched
%             X = subS;
%             [~, blobIdx]= ismember(subS(1).top{1},X(2).bottom);
%             X(2).bottom{blobIdx}  = subS(1).bottom{1};
%             X(1).replace_at = 1;
%             X(2).replace_at = 2;
%         else
%             X = [];
%         end
%     case 'conv-clean-top-unary'
%         if isequal( VAR_IN{1}, 'adjacent' )
%             X = [0];
%             return;
%         end
%         subS = VAR_IN{1};
%         is_matched = ismember( subS(1).type, 'ConvCombo-Top' );
%         if is_matched
%             X = struct([]);
%         else
%             X = [];
%         end
    otherwise
         error( 'unrecognized branch' );
end

