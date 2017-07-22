function [BLOCK_PARAM, has_caffe_net] = nldetCanonicalizeBlockParam( ...
    STAGE_NAME, BLOCK_PARAM_CUSTOM, STAGE_PARAM )

if isempty(BLOCK_PARAM_CUSTOM)
    BLOCK_PARAM_CUSTOM = struct();
end

STAGE_SPECIFIC_DIR = sysStageSpecificDir( STAGE_PARAM, STAGE_NAME, 0 );

BP = nldetBlockInitParamFromDir( STAGE_SPECIFIC_DIR );

Pdef_mc = struct();
if ~isempty( BP.mclass )
    Pdef_mc = default_eval( sprintf( '%s.default_param', BP.mclass ), struct() );
end

Pdef_cn = struct();

has_caffe_net = ~isempty(BP.net_path); % && ~BP.seal_caffe_net;

if has_caffe_net
    S = prototxt2struct( BP.net_path );
    Pdef_cn = pbPlaceHolders( S, 1 );
end

Pdef = xmerge_struct( Pdef_cn, Pdef_mc );

PH = fieldnames(Pdef);

if isempty(PH)
    BLOCK_PARAM = [];
    return;
end

BLOCK_PARAM = partial_struct( BLOCK_PARAM_CUSTOM, PH{:} );

BLOCK_PARAM = xmerge_struct( Pdef, BLOCK_PARAM );

BLOCK_PARAM = orderfields( BLOCK_PARAM );
