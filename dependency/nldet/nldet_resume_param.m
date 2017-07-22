function nldet_resume_param( train_id )

evalin( 'base', 'clear; rehash;' );

switch_pipeline_quiet nldet

evalin( 'base', ...
    sprintf( 'unstruct(sysReadParam(''Train'',%d));', train_id ) );

