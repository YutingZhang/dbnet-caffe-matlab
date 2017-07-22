function [folder_path, cleanup_handle] = get_tmp_folder( tag_str )

folder_path = '/tmp/';
if ~exist( 'tag_str', 'var' ) || isempty(tag_str)
    tag_str = 'z.matlab.tmp';
end
folder_path = fullfile(folder_path, tag_str);

PID = feature('getpid');
folder_path = [folder_path '.' int2str(PID)];

folder_path0 = folder_path;
while exist( folder_path, 'file' )
    rand_postfix = int2str(randi(2^53-1));
    folder_path = [folder_path0 '.' rand_postfix];
end

mkdir_p(folder_path);

if nargout>=2
    cleanup_handle = onCleanup( @() remove_file_recursively(folder_path) );
end
