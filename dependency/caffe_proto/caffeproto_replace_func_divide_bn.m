function X = caffeproto_replace_func_divide_bn( varargin )
% 

if strcmp( varargin{1}, 'extend' )
    X = {};
    return;
end

if strcmp( varargin{1}, 'adjacent' )
    X = [0];
    return;
end


subS = varargin{1};

bnMode = try_to_eval( 'subS.bn_param.bn_mode.val', 'LEARN' );

isValid = strcmp( subS.type{1}, 'BN' ) && ...
    ( length( subS.bottom ) == 1 ) && ...
    ( length( subS.top ) <= 1 ) && ...
    ismember( bnMode, {'LEARN','AVERAGE'} );

X = [];

if ~isValid, return; end

%
Pdef = struct();
Pdef.mass = 500;
Pdef.offline_mass = 1e6;
Pdef.always_learn = 0;

% use general param first
PARAM = scalar_struct(varargin{end}{:});
PARAM = xmerge_struct('always','always', Pdef, PARAM);


X = struct([]);

G = struct();
G.name = { [subS.name{1} '/norm'] };
G.type = {'BN'};
G.bottom = subS.bottom;
G.top    = {G.name{1}, [subS.name{1} '/batch-mean'], [subS.name{1} '/batch-var'] };
if PARAM.always_learn
    G.bn_param.bn_mode = pbEnum( 'NORM' );
else
    G.bn_param.bn_mode = pbEnum( 'AUTO_NORM' );
end
X = cat_struct( 2, X, G );

G = struct();
G.name = { [subS.name{1} '/ave'] };
G.type = {'RunningAverage'};
G.bottom = X(1).top(2:3);
G.top    = { [subS.name{1} '/mean'], [subS.name{1} '/var'] };
if PARAM.always_learn
    G.running_average_param.update_test = true;
else
    G.running_average_param.update_test = false;
end
if strcmp(bnMode,'AVERAGE')
    G.running_average_param.mass = PARAM.offline_mass;
else
    G.running_average_param.mass = PARAM.mass;
end
X = cat_struct( 2, X, G );

correctBottom = X(1).top{1};
if isfield( subS.bn_param, 'noise_param' ) && ...
        ~isempty( fieldnames( subS.bn_param.noise_param ) )
    G = struct();
    G.name = { [subS.name{1} '/add-noise'] };
    G.type = {'AddNoise'};
    G.bottom = X(1).top([1 3]);
    G.top    = { [correctBottom '/add-noise' ] };
    G.noise_param = subS.bn_param.noise_param;
    correctBottom = G.top{1};
    X = cat_struct( 2, X, G );
end

G = struct();
switch bnMode
    case 'LEARN'
        G = partial_struct(subS, 'param', 'bn_param');
        G.bn_param.noise_param = [];
        G.name = subS.name(1);
        G.type = {'BN'};
        G.bottom = [ {correctBottom}, X(2).top(1:2) ];
        G.top    = subS(1).top;
        if PARAM.always_learn
            G.bn_param.bn_mode = pbEnum( 'INFERENCE' );
        else
            G.bn_param.bn_mode = pbEnum( 'AUTO_CORRECT' );
        end
    case 'AVERAGE'
        G.name = [ subS.name{1} '/silence'];
        G.type = {'Silence'};
        G.bottom = [ {correctBottom}, X(2).top(1:2) ];
        G.top    = subS(1).top;
        X(1).name = [ subS.name{1} '/norm-side'];
        X(1).bn_param.bn_mode = pbEnum('NORM');
    otherwise
        error( 'Unrecognized bn_mode (internal error)' );
end
X = cat_struct( 2, X, G );
