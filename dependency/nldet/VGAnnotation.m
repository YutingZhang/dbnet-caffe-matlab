function S=VGAnnotation( annotationType )
% get vg annotation struct

GS = load_global_settings();
BASE_DIR = GS.VG_ANNOTATION_PATH;
SUB_FOLDERS={'v1.2'};

S = XAnnotation( BASE_DIR, SUB_FOLDERS, annotationType, @load7 );
