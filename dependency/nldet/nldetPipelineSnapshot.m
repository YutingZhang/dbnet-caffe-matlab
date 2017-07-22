function nldetPipelineSnapshot( npl, base_dir )

self_iter = npl.train_self_iter;
if ceil(self_iter)==self_iter
    self_iter_str = sprintf( '%d', self_iter );
else
    int_self_iter = floor(self_iter);
    self_iter_digit_str = sprintf( '%.2f', self_iter-int_self_iter );
    self_iter_str = sprintf( '%d.%s', int_self_iter, self_iter_digit_str(3:end) );    
end
snapshot_fn = sprintf( 'iter_%s', self_iter_str );
snapshot_dir = fullfile( base_dir, snapshot_fn );
t1 = tic_print( 'Snapshot to : %s\n Save Iter %g : ', ...
    snapshot_dir, npl.train_self_iter );
npl.save( snapshot_dir );
latest_link = fullfile( base_dir, 'latest' );
remove_file(latest_link);
create_symlink( snapshot_fn, 'latest', base_dir );
toc_print(t1);
