function [subIdxB, subS, subA, A] = caffeproto_subnet(S,varargin)
% [subIdx,subS,subA] = caffeproto_subnet(S,criteria)
% 

C = varargin;

A = [];
layerN = length(S.layer);

curMask = true( 1,layerN );

or_instead_of_and = [];
stepN = 0;
while ~isempty(C)
    stepN = stepN+1;
    T = []; argN = 0;
    if ischar(C{1})
        if ~isempty(C{1}) && C{1}(1)=='@'
            targetLayerName = C{1}(2:end);
            [~,targetLayerIdx]= ismember(targetLayerName,[S.layer.name]);
            T = false( size(curMask) );
            T(targetLayerIdx) = true;
        else
            switch C{1}
                case 'or'
                    or_instead_of_and = 1;
                    if stepN==1, curMask(:)=false; end
                case 'and'
                    or_instead_of_and = 0;
                case 'not'
                    curMask = ~curMask;
                case 'tail'
                    argN = 1;
                    A = my_aux( S, A );
                    T = caffeproto_filter_by_range(S,A,[],C{2}, 'bottom', 'top' );
                case 'head'
                    argN = 1;
                    A = my_aux( S, A );
                    T = caffeproto_filter_by_range(S,A,C{2},[], 'bottom', 'top' );
                case {'range','range:bt','range:tt','range:bb','range:tb'}
                    startAt = 'bottom';
                    endAt   = 'top';
                    switch C{1}
                        case 'range:tt', startAt='top';
                        case 'range:bb', endAt='bottom';
                        case 'range:tb', startAt='top'; endAt='bottom';
                    end
                    argN = 2;
                    A = my_aux( S, A );
                    T = caffeproto_filter_by_range(S,A,C{2},C{3},startAt,endAt);
                otherwise
                    error('unrecognized token');
            end
        end
    elseif isnumeric(C{1})
        T=false(size(curMask));
        T(C{1})=true;
    elseif islogical(C{1})
        if or_instead_of_and
            curMask = curMask | C{1};
        else
            curMask = curMask & C{1};
        end
    elseif iscell(C{1})
        T = caffeproto_subnet(S,C{1}{:});
    elseif isa(C{1}, 'function_handle')
        S_sub = S; S_sub.layer = S.layer(curMask);
        rec_idxb = caffeproto_filter_by_single_func(S_sub,C{1});
        T = curMask;
        T(curMask) = rec_idxb;
    end
    C(2:1+argN) = [];
    if isempty(T), C(1) = [];
    else C{1} = T; end
end

subIdxB=curMask;
if nargout>=2, 
    subS = S;
    subS.layer = S.layer(curMask); 
    if nargout>=3
        A = my_aux( S, A );
        subA = A;
        subA.layer = A.layer(curMask);
    end
end


function subIdxB = caffeproto_filter_by_single_func(S,filter_func)

subIdxB = false(1,length(S.layer));

for k = 1:length(S.layer)
    try
        subIdxB(k) = feval( filter_func, S.layer(k) );
    catch
        keyboard
    end
end

function subIdxB = caffeproto_filter_by_range(S,A, ...
    startToken,endToken,startAt,endAt)

layerN = length(S.layer);
layerM = length(A.layer);

startMask = false(1,layerM);
if isempty(startToken)
    startMask(:) = true;
else
    if ischar(startToken) || iscellstr(startToken) % blob name
        if strcmp( startAt, 'bottom'), B = {S.layer.bottom};
        else B = {S.layer.top}; end
        B(cellfun(@isempty,B))={''};
        p = cellfun( @(a) any(ismember(startToken,a)), B );
    else
        p = caffeproto_subnet(S,startToken);    % layer index
    end
    p = find(p);
    s = [];
    while ~isempty(p)
        s = union(s,p);
        p = unique([A.layer(p).nextLayers]);
        p = setdiff(p,s);
    end
    startMask(s) = true;
end

endMask = false(1,layerM);
if isempty(endToken)
    endMask(:) = true;
else
    if ischar(endToken) || iscellstr(endToken) % blob name
        if strcmp( endAt, 'top' ), B = {S.layer.top};
        else B={S.layer.bottom}; end
        B(cellfun(@isempty,B))={''};
        p = cellfun( @(a) any(ismember(endToken,a)), B );
    else
        p = caffeproto_subnet(S,endToken);  % layer index
    end
    p = find(p);
    s = [];
    while ~isempty(p)
        s = union(s,p);
        p = unique([A.layer(p).preLayers]);
        p = setdiff(p,s);
    end
    endMask(s) = true;
end

subIdxB = startMask & endMask;
subIdxB = subIdxB(1:layerN);


function A = my_aux( S, A )

if isempty(A),
    A = caffeproto_get_aux(S);
end
