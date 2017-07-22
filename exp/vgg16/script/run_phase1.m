function run_phase1( varargin )

PARAM_USER = scalar_struct( varargin{:} );

param_phase1
PARAM = xmerge_struct( PARAM, PARAM_USER );

sysRunStage('Train', PARAM, SPECIFIC_DIRS);

