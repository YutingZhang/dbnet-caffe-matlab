function compile_utils_filesystem()

CUR_DIR = pwd;

returnToCurDir = onCleanup( @() cd(CUR_DIR) );

THIS_DIR = fileparts(which(mfilename('fullpath')));
cd(THIS_DIR);

if ismac % with mac port (I don't know whether it works with homebrew or not)
    mex -O -lboost_system-mt -lboost_filesystem-mt create_symlink_.cpp
    mex -O -lboost_system-mt -lboost_filesystem-mt remove_file.cpp
    mex -O -lboost_system-mt -lboost_filesystem-mt remove_file_recursively.cpp
    mex -O -lboost_system-mt -lboost_filesystem-mt touch_file_.cpp
    mex -O -lboost_system-mt -lboost_filesystem-mt absolute_path.cpp
   
else
    Lopt = '';
    try
        T = eval('mex -O -lboost_system -lboost_filesystem create_symlink_.cpp');
        fprintf(1,'%s',T);
        T = eval('mex -O -lboost_system -lboost_filesystem remove_file.cpp');
        fprintf(1,'%s',T);
        T = eval('mex -O -lboost_system -lboost_filesystem remove_file_recursively.cpp');
        fprintf(1,'%s',T);
        T = eval('mex -O -lboost_system -lboost_filesystem touch_file_.cpp');
        fprintf(1,'%s',T);
        T = eval('mex -O -lboost_system -lboost_filesystem absolute_path.cpp');
        fprintf(1,'%s',T);
    catch e
        % if MATLAB boost_version is too old
        LP = strsplit( getenv('LIBRARY_PATH'), ':' );
        LP = LP( ~cellfun(@isempty,LP) );
        LP(2,:) = {' -L'};
        LP = LP([2 1],:);
        Lopt = cat(2, LP{:});
        eval( ['mex -O ' Lopt ' -lboost_system -lboost_filesystem create_symlink_.cpp'] );
        eval( ['mex -O ' Lopt ' -lboost_system -lboost_filesystem remove_file.cpp'] );
        eval( ['mex -O ' Lopt ' -lboost_system -lboost_filesystem remove_file_recursively.cpp'] );
        eval( ['mex -O ' Lopt ' -lboost_system -lboost_filesystem touch_file_.cpp'] );
        eval( ['mex -O ' Lopt ' -lboost_system -lboost_filesystem absolute_path.cpp'] );
    end
    
end


