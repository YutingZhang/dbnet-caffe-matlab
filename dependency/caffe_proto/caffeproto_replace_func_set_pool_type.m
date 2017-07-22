function X = caffeproto_replace_func_set_pool_type( varargin )

%
if strcmp( varargin{1}, 'extend' )
    X = {};
    return;
end

if strcmp( varargin{1}, 'adjacent' )
    X = [0];
    return;
end

X = [];

subS = varargin{1};

is_matched_pooling = strcmp( subS.type, 'Pooling' );

is_matched_combo = false;

if ~is_matched_pooling
    is_matched_combo = strcmp( subS.type, 'ConvCombo' ) && ...
        strcmp( subS.conv_combo_param.type, {'Convolution', 'FidaConv'} ) && ...
        ismember( 'Pooling', subS.conv_combo_param.aux_layers );
end

if ~is_matched_pooling && ~is_matched_combo
    return;
end

ARGS = varargin{end};

pool_type = ARGS{1};

X = subS;
X.pooling_param = partial_struct( default_eval( 'subS.pooling_param', struct() ), ...
    '@exclude', 'fix', 'fix_x', 'fix_y' );
X.pooling_param.pool = pbEnum( pool_type );

if strcmp( pool_type, 'FIX' )
    pool_fix = ARGS{2};
    if ~isempty( pool_fix )
        if isscalar( pool_fix )
            X.pooling_param.fix = pool_fix;
        elseif length( pool_fix ) == 2
            X.pooling_param.fix_y = pool_fix;
            X.pooling_param.fix_x = pool_fix;
        else
            error( 'Wrong pool fix location' );
        end
    end
end

if ismember( pool_type, {'AVE','MAX','STOCHASTIC'} ) && isscalar(subS.top)
    X.pooling_param = partial_struct( X.pooling_param, '@exclude', 'engine' );
end
