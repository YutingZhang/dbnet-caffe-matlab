function GS = load_global_settings()

persistent GLOBAL_SETTINGS

if isempty(GLOBAL_SETTINGS)

    script_dir = fileparts( mfilename('fullpath') );

    gs_path = fullfile(script_dir, 'global_settings.m');
    def_gs_path = fullfile(script_dir, 'global_settings_default.m');

    if ~exist( gs_path, 'file' )
        copyfile(def_gs_path, gs_path);
    end
    
    GLOBAL_SETTINGS = struct();
    
    run(gs_path);

end

GS = GLOBAL_SETTINGS;
