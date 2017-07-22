function X = caffeproto_replace_func_add_bn( varargin )


if strcmp( varargin{1}, 'extend' )
    X = { {'list', 'iterative', ...
        [{@(varargin) caffeproto_replace_func_add_bn( 'internal', varargin{1:end} )}, varargin{end}] } };
    return;
end

assert( strcmp(varargin{1},'internal'), 'wrong branch' );

VAR_IN = varargin(2:end);

if strcmp( VAR_IN{1}, 'extend' )
    X = [];
    return;
end

Pdef = struct();
Pdef.excludeTail = 1;
Pdef.bnMode = 'LEARN';
Pdef.bnAtFC = 1;


% use general param first
PARAM = scalar_struct(VAR_IN{end}{:});
PARAM = xmerge_struct('always','always', Pdef, PARAM);

if PARAM.excludeTail
    
    if strcmp( VAR_IN{1}, 'adjacent' )
        X = [0 1; 0 0];
        return;
    end

    subS = VAR_IN{1};

    isValid = strcmp( subS(1).type{1}, 'ConvCombo' ) && ...
        ~any( ismember( {'BN','BN1'}, subS(1).conv_combo_param.aux_layers ) ) && ...
        strcmp( subS(2).type{1}, 'ConvCombo' );
        
else
    
    if strcmp( VAR_IN{1}, 'adjacent' )
        X = [0];
        return;
    end

    subS = VAR_IN{1};

    isValid = strcmp( subS.type{1}, 'ConvCombo' ) && ...
        ~any( ismember( {'BN','BN1'}, subS.conv_combo_param.aux_layers ) );

end

ConvTypes = { 'Convolution', 'Deconvoltion', 'FidaConv', 'FidaDeconv' };
if isValid && ~PARAM.bnAtFC
    isValid = strcmp( subS(1).conv_combo_param.type{1}, ConvTypes );
end


X = [];

if ~isValid, return; end

X = subS(1);
insertAt = find( ismember( X.conv_combo_param.aux_layers, {'Pooling','ReLU','Dropout'} ), 1 );
if isempty( insertAt ), insertAt = 1; end

X.conv_combo_param.aux_layers = ...
    [X.conv_combo_param.aux_layers(1:insertAt-1), ...
    'BN', ...
    X.conv_combo_param.aux_layers(insertAt:end)];

X.bn_param = struct();
X.bn_param.bn_mode = pbEnum( PARAM.bnMode );
X.bn_param.scale_filler = struct( 'type', {'constant'}, 'value', 1 );
X.bn_param.shift_filler = struct( 'type', {'constant'}, 'value', 0 );


if PARAM.excludeTail
    X = cat_struct( 2, X, subS(2) );
end

