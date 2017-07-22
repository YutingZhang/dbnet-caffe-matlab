function [SNAPSHOT_ITER,TEST_ID] = nldet_test_lastest( ...
    TRAIN_ID, PRIORITY_PARAM )

evalin('base','clear');

unstruct( sysReadParam('Train',TRAIN_ID) );

depend_TrainSnapshot = sysStageMixedParam( 'TrainSnapshot' );
SNAPSHOT_ITER = depend_TrainSnapshot.TrainSnapshot_Iter;

unstruct(PRIORITY_PARAM);

pipTest

TEST_ID = regexp( STAGE_SPECIFIC_DIR, '^.*/no-([0-9]*)$','once','tokens');
TEST_ID = str2double(TEST_ID{1});
