function X = caffeproto_replace_func_combo( varargin )

if ischar(varargin{1})
    if strcmp( varargin{1}, 'adjacent' )
        X = '.*';
    elseif strcmp( varargin{1}, 'extend' )
        X = {};
    else
        error( 'Unrecognized mode' );
    end
    return;
end

subS = varargin{1};

ARGS = varargin{end};
if ~isempty(ARGS) && iscell(ARGS{1}) % subnet
    subsubIdxB = caffeproto_subnet( struct( 'layer', {subS} ), ARGS{1}{:} );
    ARGS = ARGS(2:end);
else
    subsubIdxB = true( 1, length(subS) );
end

if isempty(ARGS), 
    comboLayerName = '';
else
    comboLayerName = ARGS{1};
end
if ~isempty(subS(subsubIdxB))
    X = struct();
    BOT0 = [subS(subsubIdxB).bottom];
    TOP0 = [subS(subsubIdxB).top];
    BOT  = setdiff(BOT0,TOP0,'stable');
    TOP  = setdiff(TOP0,BOT0,'stable');
    if isempty(BOT), BOT = []; end
    if isempty(TOP), TOP = []; end
    X.name   = {comboLayerName};
    X.type   = {'Combo'};
    X.bottom = BOT;
    X.top    = TOP;
    X.bottom0 = X.bottom;
    X.top0    = X.top;
    X.layers = subS(subsubIdxB);
    
    if sum(~subsubIdxB)>0
        posRep = find( subsubIdxB, 1, 'first' );
        X.replace_at = posRep;
        X_orig = subS( ~subsubIdxB );
        replace_at_idx = num2cell(find(~subsubIdxB));
        [X_orig.replace_at] = replace_at_idx{:};
        X1 = cat_struct(2, X_orig(1:(posRep-1)), X, X_orig(posRep:end));
        X = X1;
    end
else
    X = [];
end
