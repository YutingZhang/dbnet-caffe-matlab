function less( str )

[tmp_folder, tmp_folder_cleanup] = get_tmp_folder( 'zmatlab.less' );

if ~ischar(str)
    str = any2str(str);
end

fn = fullfile( tmp_folder, 'less.txt' );
writetextfile( str, fn );

system( sprintf('less "%s"', fn) );

