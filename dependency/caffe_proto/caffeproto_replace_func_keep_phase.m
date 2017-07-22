function X = caffeproto_replace_func_keep_phase( varargin )

if strcmp( varargin{1}, 'extend' )
    X = {};
    return;
end

if strcmp( varargin{1}, 'adjacent' )
    X = [0];
    return;
end


subS = varargin{1};

targetPhases = varargin{end};
if ischar(targetPhases)
    targetPhases = {targetPhases};
end

X = [];
if isempty(targetPhases), return; end

targetPhases = cellfun(@upper,targetPhases,'UniformOutput',0);

includePhase = parse_phases( try_to_eval( 'subS.include', [] ) );
excludePhase = parse_phases( try_to_eval( 'subS.exclude', [] ) );

included = 1;
if ~isempty( includePhase )
    included = any( ismember(targetPhases, includePhase) );
end
if included && ~isempty(excludePhase)
    included = ~any( ismember(targetPhases, excludePhase) );
end

if ~included
    X = struct([]);
end

function P = parse_phases( S )

if isempty(S)
    P = {};
else
    P = cell(1,numel(S));
    for k = 1:numel(S)
        cph = try_to_eval( 'S(k).phase.val', [] );
        if ~isempty(cph)
            P{k} = cph;
        end
    end
    P(cellfun(@isempty,P)) = [];
end

