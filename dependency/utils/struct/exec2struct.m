function S = exec2struct( ARG___, TYPE___, PARAM___ )
% S = exec2struct( FILENAME )
% S = exec2struct( FILENAME, 'mfile' )
% S = exec2struct( CMD_STRING, 'inline' )
% S = exec2struct( CMD_STRING_CELL, 'inline' )

if exist('PARAM___','var') && ~isempty(PARAM___)
    assert( isstruct(PARAM___), 'PARAM___ must be a struct' );
    unstruct( PARAM___ );
end

if ~exist('TYPE___','var') || strcmp( TYPE___, 'mfile' )
    run(ARG___);
elseif strcmp( TYPE___, 'inline' )
    if iscell(ARG___)
        for CMD_ID___ = 1:numel(ARG___)
            eval(ARG___{CMD_ID___});
        end
    else
        eval(ARG___);
    end
else
    error('Unrecognized TYPE___');
end

VARIABLES___ = who;
VARIABLES___ = setdiff(VARIABLES___,{'ARG___','TYPE___','CMD_ID___','PARAM___'});
STRUCT_CMD___ = cellfun( @(a) {sprintf(' ''%s'', {%s},',a,a)}, VARIABLES___ );
STRUCT_CMD___ = cell2mat(STRUCT_CMD___.');
STRUCT_CMD___ = sprintf('struct(%s );',STRUCT_CMD___(1:end-1) );
S=eval(STRUCT_CMD___);

end
