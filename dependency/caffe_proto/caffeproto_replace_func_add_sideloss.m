function X = caffeproto_replace_func_add_sideloss( varargin )

if strcmp(varargin{1},'extend')
    X = {};
    return;
elseif strcmp(varargin{1},'adjacent')
    X = [0];
    return;
end

subS = varargin{1};


Pdef = struct();
Pdef.sidelossAt = []; % empty means at all layers
Pdef.sidelossWeight = 0.1;
Pdef.numClasses = [];
Pdef.blobLabel  = [];
Pdef.beforeAux  = 0; % before (1) / after (0) aux layers like pooling/unpooling
Pdef.sidelossGstd = 0.001;
Pdef.maxSideLossEdge = inf;

PARAM = scalar_struct(varargin{end}{:});
PARAM = xmerge_struct('always','always', Pdef, PARAM);

assert( ~isempty(PARAM.numClasses), 'numClasses cannot be empty' );
assert( ~isempty(PARAM.blobLabel), 'blobLabel cannot be empty' );

X = [];
isChosen = 0;
if strcmp( subS.type{1}, 'ConvCombo' )
    if isempty( PARAM.sidelossAt );
        isChosen = 1;
    else
        isChosen = isChosen || ...
            ( any(strcmpi('Pooling',PARAM.sidelossAt)) && ...
            ismember( 'Pooling' , subS.conv_combo_param.aux_layers) );
        isChosen = isChosen || ...
            ( any(strcmpi('Unpooling',PARAM.sidelossAt)) && ...
            ismember( 'Unpooling' , subS.conv_combo_param.aux_layers) );
    end
end

if ~isChosen, return; end

X = subS;
X.conv_combo_param = xmerge_struct( 'empty', 'always', X.conv_combo_param, ...
    struct( 'sideloss', { {} } ) );
X.conv_combo_param.sideloss{end+1} = { 'softmax', PARAM };
    %partial_struct(PARAM, ...
    %'sidelossWeight', 'numClasses', 'blobLabel', 'beforeAux') };

