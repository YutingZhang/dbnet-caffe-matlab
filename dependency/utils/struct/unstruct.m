function varargout = unstruct( S, override_policy_l, override_policy_r )
% Usage1: scalar mode
%         unstruct( S, override_policy_l, override_policy_r )
%         [APLLIED,OMITTED] = unstruct( ... );
% override_policy_l: always (default), isempty, never, [other function names or handles]
% override_policy_r: always (default), ~isempty, [other function names or handles]
% override_policy_r also works for ( R op L ) by using a "#" prefix. e.g. #gt
%                   or a single element cell array {@gt}
%
% Usage2: cell mode (Don't support override policies, always override)
%         unstruct( S, '-array' );
%

if ~exist('override_policy_l','var')
    override_policy_l = [];
end
if isempty( override_policy_l )
    override_policy_l = 'always';
end
if ~exist('override_policy_r','var')
    override_policy_r = [];
end
if isempty( override_policy_r )
    override_policy_r = 'always';
end

if ischar(S)
    S = evalin( 'caller', S );
end
assert( isstruct(S), 'S must be a struct' );

array_mode = ischar(override_policy_l) && strcmp( override_policy_l, '-array' );

if numel(S)~=1
    if ~array_mode
        error( 'Only cell mode support struct array' );
    end
end

F = fieldnames(S);

if ~array_mode
    % usage 1
    E = false(length(F),1);
    for k=1:length(F)
        E(k) = evalin( 'caller', ...
            sprintf('exist(''%s'',''var'')', F{k} ) );
    end

    L = true(length(F),1);
    if ischar(override_policy_l)
        for k=find(E.')
            L(k) = evalin( 'caller', ...
                sprintf([override_policy_l '(%s)'], F{k} ) );
        end
    else
        for k=find(E.')
            Ele_k  = evalin( 'caller', F{k} );
            L(k) = feval( override_policy_l, Ele_k );
        end
    end

    R = true(length(F),1);
    if ischar(override_policy_r)
        if ~isempty(override_policy_r) && override_policy_r(1) == '#'
            for k=find(E.')
                Ele_l  = evalin( 'caller', F{k} );
                R(k) = eval( sprintf([override_policy_r(2:end) '(S.%s,Ele_l)'],F{k}) );
            end
        else
            for k=find(E.')
                R(k) = eval( sprintf([override_policy_r '(S.%s)'],F{k}) );
            end
        end
    else
        if iscell(override_policy_r) && numel(override_policy_r) == 1
            for k=find((E).')
                Ele_l  = evalin( 'caller', F{k} );
                R(k) = feval( override_policy_r{1}, eval( sprintf('S.%s',F{k}) ), Ele_l );
            end
        else
            for k=find((E).')
                R(k) = feval( override_policy_r, eval( sprintf('S.%s',F{k}) ) );
            end
        end
    end

    I = L&R;

    availIdx = find(I);
    availIdx = reshape(availIdx,1,numel(availIdx));

    for k=availIdx
        assignin('caller', F{k}, eval(['S.' F{k}]) );
    end

    if nargout>=1
        varargout{1} = F(I);
    end
    if nargout>=2
        varargout{2} = F(~I);
    end
else
    % usage 2
    for k=1:length(F)
        assignin('caller', F{k}, reshape( eval(['{S.' F{k} '}']), size(S) ) );
    end
end

end
