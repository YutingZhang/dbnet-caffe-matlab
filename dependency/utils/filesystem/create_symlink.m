function error_code = create_symlink( SRC, DST, WORKING_DIR )

if exist('WORKING_DIR','var') && ~isempty(WORKING_DIR)
    CUR_DIR = pwd;
    try
        cd(WORKING_DIR)
    catch e
        error( ['Cannot cd to WORKING_DIR : ' WORKING_DIR] );
    end
else
    CUR_DIR = zeros(0);
end

cleaner = onCleanup( @() change_to_dir(CUR_DIR) );

if ~exist( 'DST', 'var' ) || isempty(DST)
    [~,FILE_NAME, FILE_EXT] = fileparts(SRC);
    DST = [FILE_NAME, FILE_EXT];
end

error_code = create_symlink_( SRC, DST );

end

function change_to_dir( TARGET )

if ischar(TARGET)
    cd(TARGET)
end

end

