function run_phase2( varargin )

PARAM_USER = scalar_struct( varargin{:} );

param_phase2
PARAM = xmerge_struct( PARAM, PARAM_USER );

sysRunStage('Train', PARAM, SPECIFIC_DIRS);

