function run_phase4( varargin )

PARAM_USER = scalar_struct( varargin{:} );

param_phase4
PARAM = xmerge_struct( PARAM, PARAM_USER );

sysRunStage('Test', PARAM, SPECIFIC_DIRS);
sysRunStage('TestStatistics', PARAM, SPECIFIC_DIRS);

