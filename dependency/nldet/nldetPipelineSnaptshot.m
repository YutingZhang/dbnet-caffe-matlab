function nldetPipelineSnaptshot( npl, base_dir )

% if ceil(npl.train_self_iter) == npl.train_self_iter
%     sub_dir = sprintf( 'iter_%d', npl.train_self_iter );
% else
%     int_iter = floor(npl.train_self_iter);
%     flt_iter = npl.train_self_iter-int_iter;
%     flt_iter_str = sprintf('%f', flt_iter);
%     flt_iter_str = flt_iter_str(3:end);
%     sub_dir = sprintf( 'iter_%d.%s', int_iter, flt_iter_str );
% end
sub_dir = sprintf( 'iter_%d', npl.train_self_iter );
snapshot_dir = fullfile( base_dir, sub_dir );
t1 = tic_print( 'Snapshot to : %s\n Save Iter %d : ', ...
    npl.train_self_iter );
npl.save( snapshot_dir );
toc_print(t1);
