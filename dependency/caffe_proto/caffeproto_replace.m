function [G, subMask] = caffeproto_replace(S, replace_func, varargin)
% G = caffeproto_group(S,replace_func)
% Input:
%   S - input proto struct
%   replace_func - grouping/ungrouping criterion (a function handle)
% Output:
%   G - grouped proto struct
%
% Examples: conv-relu-pool
%
%  ExtendedStrCell = replace_func( 'extend' );
%  return an empty matrix is no extending
%
%  AdjacentMatrix = replace_func( 'adjacent' );
%  sub_G = replace_func( sub_S, sub_A );
%  sub_G = [] if not match.
%

% handle filter
if ischar(replace_func) && strcmp(replace_func, 'subnet')
    subnet_criteria = varargin{1};
    if ~iscell(subnet_criteria)
        subnet_criteria = {subnet_criteria};
    end
    replace_funcs = varargin(2:end);
else
    subnet_criteria = {};
    replace_funcs = [ {replace_func}, varargin ];
end

% apply criterions sequentially
if length(replace_funcs)>1
    if strcmp( replace_funcs{1}, 'iterative' )
        G0 = S; j = 0;
        subMask = subnet_criteria;
        while true
            j = j+1;
            [G, subMask] = caffeproto_replace( G0, 'subnet', subMask, replace_funcs{2:end} );
            if isequal(G,G0), break; end
            if j>=100, error( 'reach maximum iteration' ); end
            G0 = G;
        end
    else
        [G, subMask] = caffeproto_replace(S,'subnet',subnet_criteria,replace_funcs{1});
        for k = 2:length(replace_funcs)
            [G, subMask] = caffeproto_replace(G,'subnet',subMask,replace_funcs{k});
        end
    end
    return;
end
replace_func = replace_funcs{1};
if iscell( replace_func ) && strcmp( replace_func{1}, 'list' )
    [G, subMask] = caffeproto_replace(S,'subnet',subnet_criteria,replace_func{2:end});
    return;
end

% from string criterion to func
replace_func_args = {};
if iscell(replace_func)
    replace_func_args = replace_func(2:end);
    replace_func = replace_func{1};
end
if ischar(replace_func)
    replace_func = eval( sprintf('@caffeproto_replace_func_%s;', replace_func) );
end

replace_func0 = replace_func;
replace_func  = @(varargin) replace_func0(varargin{:},replace_func_args);

% sub mask
layerN = length(S.layer);
[subMask,~,~,A] = caffeproto_subnet(S,subnet_criteria{:});
if isempty(subMask) || ~any(subMask)
    G = S;
    return;
end
Ameta = rmfield( A, 'layer' );

% handle criterion extending
EC = replace_func( 'extend' );
if ~isempty(EC)
    [G, subMask] = caffeproto_replace( S, 'subnet', subMask, EC{:} );
    return;
end

% process single criterion

%A = caffeproto_get_aux(S);
fullAdj = false(length(S.layer));
for k = 1:layerN
    curNextLayers = A.layer(k).nextLayers;
    curNextLayers(curNextLayers>layerN) = [];
    fullAdj(k, curNextLayers) = true; %(i,j)==1 means i's top to j's bottom
end

try
    [subAdj,subInIdx,subOutIdx] = replace_func( 'adjacent' );
catch
    subAdj = replace_func( 'adjacent' );
    subInIdx  = 'full';
    subOutIdx = 'full';
end
if ischar(subAdj)
    if ismember(subAdj,{'.*','^.*','.*$','^.*$'})
        subAdj = fullAdj(subMask,subMask);
    else
        % general regular expression
        error('not implemented');
    end
end

subLength = length(subAdj);
if ~subLength, 
    G=S; return;
end

if ischar(subInIdx),
    if strcmp(subInIdx,'full')
        subInIdx = 1:subLength;
    else
        error( 'unrecognized index token' );
    end
end
if ischar(subOutIdx),
    if strcmp(subOutIdx,'full')
        subOutIdx = 1:subLength;
    else
        error( 'unrecognized index token' );
    end
end


subAdj = logical( subAdj );
assert( ismatrix(subAdj) && size(subAdj,1)==size(subAdj,2), ...
    'subAdj must be square' );
% if subLength>1
%     subAdj1 = subAdj; subAdj1(1:subLength+1:end)=0;
%     assert( all(sum(abs(subAdj1),1)>0 | sum(abs(subAdj1),2).'>0), ...
%         'all node in subAdj should have at least one internal link' );
% end

layerIdList = 1:layerN;
maskSum = sum(subMask);
if maskSum<subLength
    G=S; return;
end
if maskSum==1
    if subLength<1
        G=S; return;
    else
        subCombines = layerIdList(subMask);
    end
else
    subCombines = nchoosek(layerIdList(subMask), subLength);
end
layerUsed = false( size(layerIdList) );

replaceBank = struct('layerIds',{},'subG',{});

for k = 1:size(subCombines,1)
    curC = subCombines(k,:);
    % no used layer should be involved
    if any(layerUsed(curC)), continue; end
    % at least one internal links for each layer
    if subLength>1
        cAdj = fullAdj(curC,curC); cAdj(1:subLength+1:end)=0;
        if ~all(sum(abs(cAdj),1)>0 | sum(abs(cAdj),2).'>0), continue; end
    end
    % indepth test & replace
    if 0    % use perms
        curPs = perms(curC);
    else
        curPs = curC;
    end
    for j = 1:size(curPs)
        curP = curPs(j,:);
        % no external link for intermediate layers
        if any( vec(fullAdj( curP(setdiff(1:subLength,subOutIdx)), setdiff(1:layerN,curP) )) ) || ...
                any( vec(fullAdj( setdiff(1:layerN,curP), curP(setdiff(1:subLength,subInIdx)) )) )
            continue;
        end
        % shape must be matched
        adjMatched = ( fullAdj(curP,curP) == subAdj );
        adjMatched(1:length(adjMatched)+1:end) = true;
        if all(adjMatched(:)) % if adjacent matrix matched
            subG = replace_func( S.layer(curP), A.layer(curP), Ameta );
            if isstruct(subG),
                if isempty(subG)
                    replaceBank(end+1).layerIds = curP;
                    replaceBank(end).subG = struct([]);
                else
                    if isfield(subG,'replace_at')
                        posRep = {subG.replace_at};
                        posRep(cellfun(@isempty,posRep)) = {1};
                        posRep = cell2mat(posRep);
                        assert(length(posRep)==length(subG));
                        subG = rmfield(subG,'replace_at');
                    else
                        posRep = ones(1,length(subG));
                    end
                    layerUsed(curP)=true;
                    uposRep = unique(posRep);
                    for r = 1:length(uposRep)
                        replaceBank(end+1).layerIds = curP(uposRep(r));
                        replaceBank(end).subG = subG(posRep==uposRep(r));
                    end
                    replaceBank(end).layerIds = ...
                        [replaceBank(end).layerIds, curP(setdiff(1:subLength,uposRep,'stable')) ];
                end
                break;
            end
        end
    end
end

L = num2cell(S.layer);
L( cat(2,replaceBank.layerIds) ) = {[]};
for k = 1:length(replaceBank)
    L{replaceBank(k).layerIds(1)} = ...
        reshape(replaceBank(k).subG,1,numel(replaceBank(k).subG));
end
M1 = arrayfun( @(a,b) repmat(b,size(a{1})), L, subMask, 'uniformOutput', false );
G = S;
G.layer = cat_struct( 2, L{~cellfun(@isempty,L)} );
subMask = cat(2, M1{:} );

G = caffeproto_apply_blob_revision(G);
