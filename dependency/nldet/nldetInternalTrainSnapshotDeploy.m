function nldetInternalTrainSnapshotDeploy( src, dst )

[~, src_subfolder, src_ext] = fileparts( src );
src_subfolder = [src_subfolder, src_ext];

[dst_pfolder, dst_subfolder, dst_ext] = fileparts( dst );
dst_subfolder = [dst_subfolder, dst_ext];

dst_path = fullfile( dst_pfolder, src_subfolder );
copy_folder( src, dst_path );

create_symlink( src_subfolder, dst_subfolder, dst_pfolder );

