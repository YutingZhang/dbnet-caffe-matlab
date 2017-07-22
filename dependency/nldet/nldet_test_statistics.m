function nldet_test_statistics( test_id, PRIORITY_PARAM )

evalin('base','clear');

unstruct( sysReadParam('Test',test_id) );
unstruct( PRIORITY_PARAM );

pipTestStatistics




