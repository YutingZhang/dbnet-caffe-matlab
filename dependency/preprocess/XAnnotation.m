function S = XAnnotation( BASE_DIR, SUB_FOLDERS, annotationType, loadFunc )
% get annotations from subfolders

if nargin<4,
    loadFunc = @load7;
end

fn = [];
for k = 1:length(SUB_FOLDERS)
    fn0 = fullfile( BASE_DIR, SUB_FOLDERS{k}, [annotationType, '.v7.mat'] );
    if exist(fn0,'file'), fn=fn0; break; end
end

assert( ~isempty(fn), 'Cannot find the annotation' );
S = cached_file_func( @(a) load_ann_mat(a,loadFunc), fn, 'vg-ann', 20 );

function S = load_ann_mat( fn, loadFunc )

t1 = tic_print('[ Load annotations: %s ] ',fn);

S = loadFunc(fn);

S = loaded2single_var(S);

toc_print(t1);
