function X = caffeproto_replace_func_ladder_noise( varargin )

if strcmp( varargin{1}, 'extend' )
    X = {};
    return;
end

if strcmp( varargin{1}, 'adjacent' )
    X = [0 1; 0 0];
    return;
end

subS = varargin{1};
X = [];

isValid = strcmp( subS(2).type{1},'Split:Branch' ) && ...
    isfield( subS(2), 'tag' ) && ~isempty(subS(2).tag) && ...
    strcmp( subS(2).tag{1}, 'LadderSkip' );

if ~isValid, return; end

if strcmp( subS(1).type{1}, 'AddNoise' ), return; end % already added

X = cat_struct(2,subS(1),struct(),subS(2));
X(1).replace_at = 1;
X(2).replace_at = 2;
X(3).replace_at = 2;

X(2).name   = { [ subS(2).bottom{1} '/noise' ] };
X(2).type   = { 'AddNoise' };
X(2).bottom = subS(2).bottom(1);
X(2).top    = X(2).name;
X(3).bottom = X(2).top;

X(2).noise_param = struct();
X(2).noise_param.type = {'gaussian'};
X(2).noise_param.std  = 1;

