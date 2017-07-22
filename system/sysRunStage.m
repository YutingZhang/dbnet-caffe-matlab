function sysRunStage( STAGE_NAME, PARAM, SPECIFIC_DIRS, WORKSPACE )

if nargin<4
    WORKSPACE = 'base';
end

assert( ismember(WORKSPACE, {'base','self'}), ...
    'wrong specification of WORKSPACE' );

MAIN_PARAM = PARAM;
STAGE_SPECIFIC_DIR = SPECIFIC_DIRS.(STAGE_NAME);

switch WORKSPACE
    case 'base'
        assignin('base','PARAM',PARAM);
        assignin('base','SPECIFIC_DIRS',SPECIFIC_DIRS);
        assignin('base','MAIN_PARAM',MAIN_PARAM);
        assignin('base','STAGE_SPECIFIC_DIR',STAGE_SPECIFIC_DIR);
        evalin('base',['pip' STAGE_NAME]);
    case 'self'
        eval(['pip' STAGE_NAME]);
    otherwise
        error('unrecognized WORKSPACE');
end