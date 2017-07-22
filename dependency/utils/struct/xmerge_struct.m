function [S, APPLIED___, OMITTED___] = xmerge_struct( ...
    override_policy_l___, override_policy_r___, varargin )
% Usage: [S, APPLIED, OMITTED] 
%           = xmerge_struct( override_policy_l, override_policy_r, S1, S2, ... )
%        [S, APPLIED, OMITTED] 
%           = xmerge_struct( override_policy_l, S1, S2, ... ) % override_policy_r = 'always'
%        [S, APPLIED, OMITTED] 
%           = xmerge_struct( S1, S2, ... ) % override_policy_l&r = 'always'
%
%        ... = xmerge_struct( ... , '-sorted' );  % use sorted order (default)
%        ... = xmerge_struct( ... , '-stable' );  % use stable order
%        ... = xmerge_struct( ... , '-rstable' ); % use reverse stable order
%
%        ... = xmerge_struct( ... , '-display' );
%        ... = xmerge_struct( ... , '-display', GROUP_NAMES );
%        ... = xmerge_struct( ... , '-display', GROUP_NAME1, GROUP_NAME2, ... );
%        ... = xmerge_struct( ... , '-display', ..., '-serialize_func', serialize_func );
% override_policy_l, override_policy_r: refer to unstruct
% Suppose the inputs are S1, S2, ..., Sn, 
%   Without -array, APPLIED, OMITTED are both n*1 cell array of string cell arrays;
%   With -array, each cell of APPLIED, OMITTED are as above.

if isstruct( override_policy_l___ )
    if nargin==1
        VARARGIN = {override_policy_l___};
    else
        VARARGIN = [{override_policy_l___, override_policy_r___ }, varargin];
    end
    override_policy_l___ = 'always';
    override_policy_r___ = 'always';
elseif isstruct( override_policy_r___ )
    VARARGIN = [{override_policy_r___}, varargin];
    override_policy_r___ = 'always';
else
    VARARGIN = varargin;
end

GROUP_NUM___ = 0;
for INPUT_IDX___ = 1:length(VARARGIN)
    if (ischar(VARARGIN{INPUT_IDX___}) && ~isempty(VARARGIN{INPUT_IDX___}) && ...
            VARARGIN{INPUT_IDX___}(1) == '-')
        break;
    end
    GROUP_NUM___ = INPUT_IDX___;
end

OPTIONS___ = VARARGIN( GROUP_NUM___+1:end );

CHAR_OPTIONS_IDXB___ = cellfun(@ischar,OPTIONS___);

INPUT_NUMEL___ = zeros(GROUP_NUM___,1);
for INPUT_IDX___ = 1:GROUP_NUM___
    if ischar(VARARGIN{INPUT_IDX___})
        INPUT_NUMEL___(INPUT_IDX___) = evalin( 'caller', ...
            ['numel(' VARARGIN{INPUT_IDX___} ')'] );
    else
        INPUT_NUMEL___(INPUT_IDX___) = numel( VARARGIN{INPUT_IDX___} );
    end
end

if ~all(INPUT_NUMEL___==1)
    error( 'Donnot support struct array' );
end

IS_SORTED___ = ismember( '-sorted', OPTIONS___(CHAR_OPTIONS_IDXB___) );
IS_STABLE___ = ismember( '-stable', OPTIONS___(CHAR_OPTIONS_IDXB___) );
IS_REVERSE_STABLE___ = ismember( '-rstable', OPTIONS___(CHAR_OPTIONS_IDXB___) );

if sum([IS_SORTED___,IS_STABLE___,IS_REVERSE_STABLE___])>1
    error( 'Conflict options. Only one of -sorted, -stable, -rstable can be specified' );
end

FIELD_NAMES___ = cell(1,GROUP_NUM___);
for INPUT_IDX___ = 1:GROUP_NUM___
    if ischar(VARARGIN{INPUT_IDX___})
        FIELD_NAMES___{INPUT_IDX___} = evalin( 'caller', ...
            ['fieldnames(' VARARGIN{INPUT_IDX___} ')'] );
    else
        FIELD_NAMES___{INPUT_IDX___} = fieldnames( VARARGIN{INPUT_IDX___} );
    end
end

if IS_REVERSE_STABLE___
    FINAL_ORDER___ = ordered_union( 'rstable', FIELD_NAMES___{:} );
elseif IS_STABLE___
    FINAL_ORDER___ = ordered_union( 'stable', FIELD_NAMES___{:} );
else % IS_SORTED___ is the default one
    FINAL_ORDER___ = ordered_union( 'sorted', FIELD_NAMES___{:} );
end


APPLIED___ = cell(GROUP_NUM___,1);
OMITTED___ = cell(GROUP_NUM___,1);

for INPUT_IDX___ = 1:GROUP_NUM___
    if ischar(VARARGIN{INPUT_IDX___})
        [APPLIED___{INPUT_IDX___}, OMITTED___{INPUT_IDX___} ] = ...
            unstruct( evalin( 'caller', VARARGIN{INPUT_IDX___} ), ...
            override_policy_l___, override_policy_r___ );
    elseif isstruct(VARARGIN{INPUT_IDX___})
        [APPLIED___{INPUT_IDX___}, OMITTED___{INPUT_IDX___} ] = ...
            unstruct( VARARGIN{INPUT_IDX___}, ...
            override_policy_l___, override_policy_r___ );
    else
        error( 'inputs should be either struct or string' );
    end
end

CUR_APPLIED___ = {};
for INPUT_IDX___ = GROUP_NUM___:-1:1
    CUR_REMOVE___ = intersect( APPLIED___{INPUT_IDX___}, CUR_APPLIED___ );
    OMITTED___{INPUT_IDX___} = [ OMITTED___{INPUT_IDX___}; CUR_REMOVE___ ];
    APPLIED___{INPUT_IDX___} = setdiff( APPLIED___{INPUT_IDX___}, CUR_REMOVE___ );
    CUR_APPLIED___ = [ APPLIED___{INPUT_IDX___}; CUR_APPLIED___ ];
end

S = var2struct( FINAL_ORDER___{:} );

% No need to avoid naming confliction below

CHAR_OPTIONS_IDX___ = find( CHAR_OPTIONS_IDXB___ );

[need2display, display_token] = ismember( '-display', OPTIONS___(CHAR_OPTIONS_IDXB___) );
if need2display
    display_token = CHAR_OPTIONS_IDX___(display_token);
    
    group_names = {};
    if length(OPTIONS___)>=display_token+1
        if iscell(OPTIONS___{display_token+1})
            group_names = OPTIONS___{display_token+1};
        else
            r = display_token;
            for k=display_token+1:length(OPTIONS___)
                if ~ischar(OPTIONS___{k}) || OPTIONS___{k}(1) == '-'
                    break;
                end
                r = k;
            end
            group_names = OPTIONS___(display_token+1:r);
        end
    end
    
    serialize_func = [];
    [has_serialize_func, serialize_func_token] = ismember( '-serialize_func', OPTIONS___(CHAR_OPTIONS_IDXB___) );
    if has_serialize_func
        serialize_func_token = CHAR_OPTIONS_IDX___(serialize_func_token);
        if length(OPTIONS___) < serialize_func_token+1
            error('No serialize_func is given');
        end
        serialize_func = OPTIONS___{serialize_func_token+1};
    end
    
    display_grouped_vars( serialize_func, S, group_names, APPLIED___{:} );
end

end
