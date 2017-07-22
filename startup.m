function startup

script_dir = fileparts( mfilename('fullpath') );

addpath( genpath( fullfile(script_dir, 'dependency') ) );
addpath( genpath( fullfile(script_dir, 'pipeline') ) );
addpath( genpath( fullfile(script_dir, 'system') ) );
addpath( fullfile(script_dir, 'caffe/matlab') );

