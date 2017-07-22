function varargout = xunstruct( ...
    override_policy_l___, override_policy_r___, varargin )
% Usage: xunstruct( override_policy_l, override_policy_r, S2, S3, ..., OPTIONS )
%        [APPLIED, OMITTED] = xunstruct(...)
% override_policy_l, override_policy_r: refer to unstruct
% OPTIONS: refer to xmerge_struct
% Suppose the inputs are S2, S3, ..., Sn, 
%        APPLIED, OMITTED are both n*1 cell array of string cell arrays

GROUP_NUM___ = 0;
for INPUT_IDX___ = 1:length(varargin)
    if ischar(varargin{INPUT_IDX___})
        if ~isempty(varargin{INPUT_IDX___}) && ...
                varargin{INPUT_IDX___}(1) == '-'
            break;
        end
        varargin{INPUT_IDX___} = evalin( 'caller', varargin{INPUT_IDX___} );
    end
    GROUP_NUM___ = INPUT_IDX___;
end

INPUT_NUMEL___ = cellfun( @numel, varargin(1:GROUP_NUM___) );
if ~all(INPUT_NUMEL___==1)
    error( 'Donnot support struct array' );
end


FIELD_NAMES___ = cell(1,GROUP_NUM___);
for INPUT_IDX___ = 1:GROUP_NUM___
	FIELD_NAMES___{INPUT_IDX___} = fieldnames( varargin{INPUT_IDX___} );
end

OPTIONS___ = varargin( GROUP_NUM___+1:end );

CHAR_OPTIONS_IDXB___ = cellfun(@ischar,OPTIONS___);

IS_SORTED___ = ismember( '-sorted', OPTIONS___(CHAR_OPTIONS_IDXB___) );
IS_STABLE___ = ismember( '-stable', OPTIONS___(CHAR_OPTIONS_IDXB___) );
IS_REVERSE_STABLE___ = ismember( '-rstable', OPTIONS___(CHAR_OPTIONS_IDXB___) );

if sum([IS_SORTED___,IS_STABLE___,IS_REVERSE_STABLE___])>1
    error( 'Conflict options. Only one of -sorted, -stable, -rstable can be specified' );
end

if IS_REVERSE_STABLE___
    F1 = ordered_union( 'rstable', FIELD_NAMES___{:} );
elseif IS_STABLE___
    F1 = ordered_union( 'stable', FIELD_NAMES___{:} );
else % IS_SORTED___ is the default for pre-sorting
    F1 = ordered_union( 'sorted', FIELD_NAMES___{:} );
end

C = cellfun( @(a) {['''' a ''',']}, F1.' );
C = cat(2,C{:});
C = sprintf('var2struct(%s,''-ignore'')',C(1:end-1));
S1 = evalin( 'caller', C );

[S, APPLIED___, OMITTED___] = xmerge_struct( ...
    override_policy_l___, override_policy_r___, S1, varargin{:} );

if nargout>=1
    varargout{1} = APPLIED___;
end
if nargout>=2
    varargout{2} = OMITTED___;
end

assignin( 'caller', 'xunstruct_TMP_STRUCT_FOR_EXPENSION___', S ); 
[~] = evalin( 'caller', 'unstruct(xunstruct_TMP_STRUCT_FOR_EXPENSION___)' );
evalin( 'caller', 'clear xunstruct_TMP_STRUCT_FOR_EXPENSION___' );

end
