function run_phase3( varargin )

PARAM_USER = scalar_struct( varargin{:} );

param_phase3
PARAM = xmerge_struct( PARAM, PARAM_USER );

sysRunStage('Train', PARAM, SPECIFIC_DIRS);

