function Compile_RunLength

RunLength_Folder = fileparts(mfilename('fullpath'));
CUR_DIR = pwd;
back2pwd = onCleanup( @() cd(CUR_DIR) );

cd(RunLength_Folder)
mex -O RunLength.cpp
