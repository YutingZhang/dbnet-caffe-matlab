function [X,varargout] = caffeproto_replace_func_upgrade( varargin )

if strcmp(varargin{1},'extend')
    X = {};
    return;
elseif strcmp(varargin{1},'adjacent')
    X = [0];
    return;
end

SNameMap = {
    'LRN',  'LRN'
    'BN',   'BN'
    'RELU', 'ReLU'
    'SOFTMAX_LOSS', 'SoftmaxWithLoss'
    };

subS = varargin{1};

X = subS;

if ~isempty( X.type )
    if ~iscell(X.type)
        t = X.type(1).val();
        [is_special_map, special_map_idx] = ...
            ismember(t,SNameMap(:,1));
        if is_special_map
            t1 = SNameMap{special_map_idx,2};
        else
            t1 = lower( t );
            t1 = [ '_' t1 ];
            upidx=find(t1=='_')+1;
            upidx(upidx>length(t1))=[];
            t1(upidx) = upper(t1(upidx));
            t1(upidx-1) = [];
        end
        X.type = {t1};
    end
end

if isfield( X, 'blobs_lr' )
    for k = 1:length(X.blobs_lr)
        X.param(k).lr_mult = X.blobs_lr(k);
    end
    X = rmfield( X, 'blobs_lr' );
end

if isfield( X, 'weight_decay' )
    for k = 1:length(X.weight_decay)
        X.param(k).decay_mult = X.weight_decay(k);
    end
    X = rmfield( X, 'weight_decay' );
end

if strcmp(X.type{1},'Unpooling')
    X.type{1} = 'Depooling';
    if isfield( X, 'unpooling_param' ) 
        X = rename_field( X, 'unpooling_param', 'pooling_param' );
        if isfield( X.pooling_param, 'unpool' )
            X.pooling_param = rename_field( X.pooling_param, 'unpool', 'pool' ); 
        end
        X0 = X;
        X.pooling_param = partial_struct( X0.pooling_param, '@exclude', 'unpool_.*' );
        if ~isequal(X,X0)
            warning( 'unpool specific field(s) is removed' );
        end
    end
elseif strcmp(X.type{1},'EltwiseAccuracy')
    X.type{1} = 'Accuracy';
    if isfield( X, 'eltwise_accuracy_param' )
        X = rename_field(X,'eltwise_accuracy_param','accuracy_param');
    end
end

if isequal(X,subS), X = []; end

