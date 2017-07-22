function [X, varargout] = caffeproto_replace_func_convcombo( varargin )

ARGS = varargin{end};

if ischar(varargin{1})
    if strcmp(varargin{1},'extend')
        X = { 
            [ {@(varargin) caffeproto_replace_func_convcombo( 'conv', varargin{:} )}, ARGS ], ...
            { 'list', 'iterative', ...
                [ {@(varargin) caffeproto_replace_func_convcombo( 'conv-relu', varargin{:} )}, ARGS ], ...
                [ {@(varargin) caffeproto_replace_func_convcombo( 'conv-pooling', varargin{:} )}, ARGS ], ...
                [ {@(varargin) caffeproto_replace_func_convcombo( 'unpooling-conv', varargin{:} )}, ARGS ], ...
                [ {@(varargin) caffeproto_replace_func_convcombo( 'conv-lrn', varargin{:} )}, ARGS ], ...
                [ {@(varargin) caffeproto_replace_func_convcombo( 'conv-bn', varargin{:} )}, ARGS ], ...
                [ {@(varargin) caffeproto_replace_func_convcombo( 'dropout-conv', varargin{:} )}, ARGS ] 
            } , ...
            [ {@(varargin) caffeproto_replace_func_convcombo( 'convcombo-add-top-split', varargin{:} ) }, ARGS ], ...
            [ {@(varargin) caffeproto_replace_func_remove_split( varargin{:} ) }, {'extra_condition', ...
            @(subS) try_to_eval('subS.origin_from_convcombo',false) } ]
            };
        return;
    end
end

VAR_IN = varargin(2:end);

if ~isempty(VAR_IN) && strcmp( VAR_IN{1}, 'extend' )
    X = {};
    return;
end

ConvTypes = {'Convolution','Deconvolution','FidaConv','FidaDeconv', ...
    'InnerProduct','Eltwise', 'Pooling' };

Pdef = struct();
Pdef.keepAuxLayerName = false;

PARAM=scalar_struct(VAR_IN{end}{:});
PARAM = xmerge_struct('always','always', Pdef, PARAM);

switch varargin{1}
    case 'conv'
        if isequal( VAR_IN{1}, 'adjacent' )
            X = [0];
            varargout = {[1],[1]};
        else
            subS = VAR_IN{1};
            is_matched = ismember( subS(1).type, ConvTypes );
            if is_matched
                if strcmp( subS(1).type, 'Pooling' ) && ...
                    ~default_eval( 'subS(1).pooling_param.global_pooling', false )
                    X=[];
                %elseif strcmp(subS(1).type,'Eltwise') && ~isscalar(subS(1).bottom)
                %    X=[];
                else
                    X = caffeproto_basic_convcombo(subS(1));
                end
            else
                X = [];
            end
        end
    case 'conv-relu'
        if isequal( VAR_IN{1}, 'adjacent' )
            X = [0 1; 0 0];
            varargout = {[1],[2]};
        else
            subS = VAR_IN{1};
            is_matched = ismember( subS(1).type, 'ConvCombo' ) && ...
                strcmp( subS(2).type, 'ReLU' ) && ...
                ~ismember('ReLU',subS(1).conv_combo_param.aux_layers);
            if is_matched
                X = subS(1);
                X.conv_combo_param.aux_layers{end+1} = 'ReLU';
                relu_param = default_eval( 'subS(2).relu_param', [] );
                if ~isempty(X)
                    X.relu_param = relu_param;
                end
                X.top = subS(2).top;
                if PARAM.keepAuxLayerName
                    X = caffeproto_convcombo_set_aux_name( ...
                        X,'ReLU',subS(2).name{1});
                end
            else
                X = [];
            end
        end
    case 'conv-pooling'
        if isequal( VAR_IN{1}, 'adjacent' )
            X = [0 1; 0 0];
            varargout = {[1],[2]};
        else
            subS = VAR_IN{1};
            is_matched = ismember( subS(1).type, 'ConvCombo' ) && ...
                ~strcmp( subS(1).conv_combo_param.type, 'Pooling' ) && ...
                strcmp( subS(2).type, 'Pooling' ) && ...
                ~any(ismember({'Pooling','Unpooling'},subS(1).conv_combo_param.aux_layers));
            is_matched = is_matched && ~default_eval( 'subS(2).pooling_param.global_pooling', false );
            if is_matched
                X = subS(1);
                X.conv_combo_param.aux_layers{end+1} = 'Pooling';
                X = transfer_field( X, subS(2), 'pooling_param' );
                if isfield( subS, 'propagate_down' ) && ...
                        numel(subS(2).propagate_down)>=2
                    X.conv_combo_param.pooling_backprop_switch = ...
                        subS(2).propagate_down(2);
                end
                X.top = subS(2).top(1);
                if length(subS(2).top)>1
                    X = caffeproto_convcombo_set_aux_blobs( ...
                        X,'Pooling',[],subS(2).top(2:end));
                end
                if PARAM.keepAuxLayerName
                    X = caffeproto_convcombo_set_aux_name( ...
                        X,'Pooling',subS(2).name{1});
                end
            else
                X = [];
            end
            top_shape0 = caffeproto_avail_ioshape( X, 't', 1 );
            top_shape1 = caffeproto_avail_ioshape( subS(2), 't', 1 );
            if ~isempty(top_shape0)
                if ~isempty(top_shape1)
                    X.aux.top_shapes = {top_shape1};
                else
                    error('missing top shape');
                end
            end
        end
    case 'conv-lrn'
        if isequal( VAR_IN{1}, 'adjacent' )
            X = [0 1; 0 0];
            varargout = {[1],[2]};
        else
            subS = VAR_IN{1};
            is_matched = ismember( subS(1).type, 'ConvCombo' ) && ...
                strcmp( subS(2).type, 'LRN' ) && ...
                ~ismember('LRN',subS(1).conv_combo_param.aux_layers);
            if is_matched
                X = subS(1);
                X.conv_combo_param.aux_layers{end+1} = 'LRN';
                X = transfer_field( X, subS(2), 'lrn_param' );
                X.top = subS(2).top;
                if PARAM.keepAuxLayerName
                    X = caffeproto_convcombo_set_aux_name( ...
                        X,'LRN',subS(2).name{1});
                end
            else
                X = [];
            end
        end
    case 'conv-bn'
        if isequal( VAR_IN{1}, 'adjacent' )
            X = [0 1; 0 0];
            varargout = {[1],[2]};
        else
            subS = VAR_IN{1};
            is_matched = ismember( subS(1).type, 'ConvCombo' ) && ...
                strcmp( subS(2).type, 'BN' ) && ...
                ~ismember('BN',subS(1).conv_combo_param.aux_layers);
            if is_matched
                X = subS(1);
                X.conv_combo_param.aux_layers{end+1} = 'BN';
                X = transfer_field( X, subS(2), 'bn_param' );
                if isfield( subS(2), 'param' );
                    X.bn_param.param = subS(2).param;
                end
                X.top = subS(2).top;
                if length(subS(2).top)>1
                    X = caffeproto_convcombo_set_aux_blobs( ...
                        X,'BN',[],subS(2).top(2:end));
                end
                if PARAM.keepAuxLayerName
                    X = caffeproto_convcombo_set_aux_name( ...
                        X,'BN',subS(2).name{1});
                end
                if isequal(subS(2).bottom, subS(2).top)
                    X.bn_param.inplace = 1;
                end
            else
                X = [];
            end
        end
    case 'dropout-conv'
        if isequal( VAR_IN{1}, 'adjacent' )
            X = [0 1; 0 0];
            varargout = {[1],[2]};
        else
            subS = VAR_IN{1};
            is_matched = ismember( subS(2).type, 'ConvCombo' ) && ...
                strcmp( subS(1).type, 'Dropout' ) && ...
                ~ismember('Dropout',subS(2).conv_combo_param.aux_layers);
            if is_matched
                X = subS(2);
                X.conv_combo_param.aux_layers{end+1} = 'Dropout';
                X = transfer_field( X, subS(1), 'dropout_param' );
                X.bottom = subS(1).bottom;
                if PARAM.keepAuxLayerName
                    X = caffeproto_convcombo_set_aux_name( ...
                        X,'Dropout',subS(1).name{1});
                end
            else
                X = [];
            end
        end
    case 'unpooling-conv'
        if isequal( VAR_IN{1}, 'adjacent' )
            X = [0 1; 0 0];
            varargout = {[1],[2]};
        else
            subS = VAR_IN{1};
            is_matched = ismember( subS(2).type, 'ConvCombo' ) && ...
                strcmp( subS(1).type, 'Depooling' ) && ...
                ~any(ismember({'Pooling','Unpooling'},subS(2).conv_combo_param.aux_layers));
            if is_matched
                X = subS(2);
                X.conv_combo_param.aux_layers{end+1} = 'Unpooling';
                X = transfer_field( X, subS(1), 'pooling_param' );
                X.bottom = subS(1).bottom(1);
                if length(subS(1).bottom)>1
                    X = caffeproto_convcombo_set_aux_blobs( ...
                        X,'Unpooling',subS(1).bottom(2:end),[]);
                end
                if PARAM.keepAuxLayerName
                    X = caffeproto_convcombo_set_aux_name( ...
                        X,'Unpooling',subS(1).name{1});
                end
            else
                X = [];
            end
        end
    case 'convcombo-add-top-split'
        if isequal( VAR_IN{1}, 'adjacent' )
            X = [0];
        else
            subS = VAR_IN{1};
            is_matched = ismember( subS.type, 'ConvCombo' ) && ...
                ~isempty( try_to_eval( 'subS.conv_combo_param.top', [] ) );
            if is_matched
                X = subS;
                X.conv_combo_param = rmfield(subS.conv_combo_param,'top');
                X.top(1) = subS(1).conv_combo_param.top;
                G = struct();
                G.name = { [subS.name{1} '/top-split'] };
                G.type = { 'Split:Branch' };
                G.bottom = X.top(1);
                G.top    = subS.top(1);
                G.origin_from_convcombo = true;
                X = cat_struct(2,X,G);
            else
                X = [];
            end
        end
    otherwise
        error( 'unrecognized branch' );
end

