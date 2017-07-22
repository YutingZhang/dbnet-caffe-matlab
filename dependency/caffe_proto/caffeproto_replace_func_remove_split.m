function [X, varargout] = caffeproto_replace_func_remove_split( varargin )

if strcmp(varargin{1},'extend')
    X = { { 'list', 'iterative', ...
        @(varargin) caffeproto_replace_func_remove_split( 'inplace', varargin{:} ), ...
        @(varargin) caffeproto_replace_func_remove_split( 'linked', varargin{:} ) }, ...
        @(varargin) caffeproto_replace_func_remove_split( 'orphan', varargin{:} ), ...
        @(varargin) caffeproto_replace_func_remove_split( 'reset-bottom', varargin{:} ) };
    return;
end

VAR_IN = varargin(2:end);

if strcmp(VAR_IN{1},'extend')
    X = {};
    return;
end

Pdef = struct();
Pdef.extra_condition = @(subS) true;

PARAM = scalar_struct(varargin{end}{:});
if ~isstruct(PARAM), PARAM = struct( 'default', {PARAM} ); end
PARAM = xmerge_struct('always','always', Pdef, PARAM);

X = [];

switch varargin{1}
    case 'inplace'
        if strcmp(VAR_IN{1},'adjacent')
            X = [0];
            % varargout = {[1],[]};
            return;
        end
        subS = VAR_IN{1};
        X = [];
        if ~ismember( subS(1).type{1}, {'Split','Split:Branch'} ) || ...
                ~PARAM.extra_condition(subS(1))
            return;
        end
        if isempty( subS.top )
            X = struct([]);
        else
            inplace2cleanIdxb = strcmp( subS.top, subS.bottom );
            if any(inplace2cleanIdxb)
                X = subS;
                X.top(inplace2cleanIdxb) = [];
                if isempty( subS.top ), X = struct([]); end
            end
        end
    case 'linked'
        if strcmp(VAR_IN{1},'adjacent')
            X = [0 1; 0 0];
            return;
        end

        subS = VAR_IN{1};
        subA = VAR_IN{2};
        if ~ismember( subS(1).type{1}, {'Split','Split:Branch'} ) || ...
                ~PARAM.extra_condition(subS(1))
            return;
        end

        %subA  = varargin{2};
        %subArel = caffeproto_canonicalize_subaux_idx(subA);

        npb = try_to_eval( 'subS(2).nonprefixable_bottom', [] );        
        X     = subS;
        bidxb0 = ismember( X(2).bottom, subS(1).top );
        bidxb  = bidxb0; bidxb(npb) = false;
        npIdxb = xor(bidxb0,bidxb);
        
        if ~isempty(bidxb)
            if strcmp(subS(1).type{1}, 'Split') && length(subS(2).bottom)>1
                is_next_inplace=any( ismember( subS(2).bottom( bidxb ), subS(2).top ) );
                if is_next_inplace
                    X = [];
                    return;
                end
            end
            X(2).bottom( bidxb ) = subS(1).bottom(1);
            
            %tidxb = ismember( X(2).top, subS(1).top ); % For inplace layers
            %X(2).top( tidxb ) = subS(1).bottom(1); % how about two inplace layers?? potential bugs
            
            linkedTopIdxB  = ismember( X(1).top, subS(2).bottom );
            tocleanTopIdxB = linkedTopIdxB;
            tocleanTopIdxB(linkedTopIdxB) = cellfun( @(a) size(a,2), ...
                subA(1).nextBlobs(linkedTopIdxB) )<=1;
            X(1).top(tocleanTopIdxB) = [];
            
            % X(1).type{1} = 'Split:Branch';
            X(1).replace_at = 1;
            X(2).replace_at = 2;
        end
        if any( npIdxb ) % handle non-prefixable
            if ~isfield( X, 'bottom_split_cache' ) || ...
                    isempty( X(2).bottom_split_cache )
                X(2).bottom_split_cache = cell( size(X(2).bottom) );
            end
            X(2).bottom_split_cache( npIdxb ) = X(2).bottom( npIdxb );
            X(2).bottom( npIdxb ) = cellfun( @(a) ['!!!!!split-cached:' a], ...
                X(2).bottom_split_cache( npIdxb ), 'UniformOutput', 0);
        end
    case 'orphan'
        if strcmp(VAR_IN{1},'adjacent')
            X = [0];
            % varargout = {[1],[]};
            return;
        end
        
        subS = VAR_IN{1};
        subA  = VAR_IN{2};
        Ameta = VAR_IN{3};
        
        if ~ismember( subS(1).type{1}, {'Split','Split:Branch'} ) || ...
                ~PARAM.extra_condition(subS(1))
            return;
        end
        
        if isempty( subS(1).top )
            X = struct([]);
        elseif ismember( subS(1).type{1}, {'Split:Branch'} )
            X = struct([]);
%             isSafeToRemove = false(size(subA.nextBlobs));
%             for j = 1:numel(subA.nextBlobs)
%                 isSafeToRemove(j) = all( subA.nextBlobs{j}(1,:) == Ameta.topLayerIdx );
%             end
%             if all(isSafeToRemove),
%                 X = struct([]);
%             else
%                 X = subS;
%                 X.top(isSafeToRemove) = [];
%             end
        elseif length( subS(1).top ) == 1
            X = partial_struct( subS, 'name', 'bottom', 'top' );
            X.type = {'Reshape'}; % more memory efficient
            X.reshape_param = struct( 'num_axes', 0 );
        end
        
    case 'reset-bottom'
        if strcmp(VAR_IN{1},'adjacent')
            X = [0];
            % varargout = {[1],[]};
            return;
        end
        
        subS = VAR_IN{1};
        X = [];
        if isempty( try_to_eval( 'subS.bottom_split_cache', [] ) )
            return;
        end
        X = subS;
        npIdxb = ~cellfun( @isempty, subS.bottom_split_cache );
        X.bottom( npIdxb ) = X.bottom_split_cache( npIdxb );
        X = rmfield( X, 'bottom_split_cache' );
    otherwise
        error( 'unrecognized branch' );
end
        
