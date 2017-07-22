function X = caffeproto_replace_func_set_cudnn( varargin )

if strcmp( varargin{1}, 'extend' )
    X = {};
    return;
end


if strcmp( varargin{1}, 'adjacent' )
    X = [0];
    return;
end

tCudnn = pbEnum('CUDNN');
tCaffe = pbEnum('CAFFE');

ARGS = varargin{end};
if isempty( ARGS )
    engineType = [];
elseif boolean( ARGS{1} )
    engineType = tCudnn;
else
    engineType = tCaffe;
end

subS = varargin{1};

X = subS;
switch subS.type{1}
    case {'Convolution','Deconvolution','FidaConv','FidaDeconv'}
        X.convolution_param.engine = engineType;
    case 'Pooling'
        if ismember( default_eval( 'X.pooling_param.pool.val', 'MAX' ), {'MAX','AVE','STOCHASTIC'} )
            X.pooling_param.engine = engineType;
        else
            X.pooling_param.engine = tCaffe;
        end
    case 'Depooling'
        X.pooling_param.engine = tCaffe;
    case {'Softmax','SoftmaxWithLoss'}
        X.softmax_param.engine = engineType;
    case {'LRN'}
        X.lrn_param.engine = engineType;
    otherwise
        X = [];
end

