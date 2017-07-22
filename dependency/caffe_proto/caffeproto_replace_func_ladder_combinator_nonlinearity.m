function [X, varargout] = caffeproto_replace_func_ladder_combinator_nonlinearity ( ...
    varargin )


ARGS = varargin{end};

if strcmp(varargin{1},'extend')
    X = { 
        [ {@(varargin) caffeproto_replace_func_ladder_combinator_nonlinearity( 'two-bottom', varargin{:} )}, ARGS ] %, ...
%         [ {@(varargin) caffeproto_replace_func_ladder_combinator_nonlinearity( 'single-bottom', varargin{:} )}, ARGS ]
        };
    return;
end

assert(ischar(varargin{1}), 'wrong branch token');

VAR_IN = varargin(2:end);
if strcmp(VAR_IN{1},'extend')
    X = {};
    return;
end

X = [];
switch varargin{1}
    case 'two-bottom'
        if strcmp(VAR_IN{1},'adjacent')
            X = [
                0 0 1
                0 0 1
                0 0 0];
            return;
        end
        subS = VAR_IN{1};
        subA = VAR_IN{2};
%         isValid = ismember( subS(1).type{1}, 'ConvCombo' ) && ...
%             strcmp( subS(2).type{1}, 'ConvCombo' ) && ...
%             strcmp( subS(3).type{1}, 'LadderCombinator0' );
        isValid = any( strcmp( [subS(1).type(1),subS(2).type(1)], 'ConvCombo' ) ) && ...
            strcmp( subS(3).type{1}, 'LadderCombinator0' );
        if ~isValid, return; end
        subArel = caffeproto_canonicalize_subaux_idx(subA);
        idx_topdown = subArel(3).preBlobs{1}(1);
        idx_skip    = subArel(3).preBlobs{2}(1);
        idx_ladder  = 3;
        X = subS([1,2]);
        X(1).replace_at = 1;
        X(2).replace_at = 2;
%     case 'single-bottom'
%         if strcmp(VAR_IN{1},'adjacent')
%             X = [
%                 0 1
%                 0 0 ];
%             return;
%         end
%         subS = VAR_IN{1};
%         isValid = strcmp( subS(1).type{1}, 'ConvCombo' ) && ...
%             strcmp( subS(2).type{1}, 'LadderCombinator0' );
%         if ~isValid, return; end
%         idx_topdown = 1;
%         idx_skip    = 1;
%         idx_ladder  = 2;
%         X(1).replace_at = 1;
    otherwise
        error('unrecognized branch token');
end

% bottom 1 is for higher-level decoder (top-down)
% bottom 2 is for encoder (skip-link)

% for skip-link stream, put a branching tag
ladderSkipName = [ subS(idx_ladder).bottom{2} '/ladder-skip' ];
if strcmp( X(idx_skip).type{1}, 'ConvCombo' )
    alN = length( X(idx_skip).conv_combo_param.aux_layers );
    hasSingleEltPool = 0;
    
    nlIdx = alN+1;
    if alN>0
        endAXType = X(idx_skip).conv_combo_param.aux_layers{alN};
        if strcmp( endAXType, 'Pooling' )
            nlIdx = alN;
            if ismember( X(idx_skip).pooling_param.pool(1).val(), {'MAX','SWITCH'} )
                if alN>1
                    previousAXType = X(idx_skip).conv_combo_param.aux_layers{alN-1};
                    if ismember( previousAXType, 'ReLU' )
                        X(idx_skip).conv_combo_param.aux_layers(end-1:end) = ...
                            X(idx_skip).conv_combo_param.aux_layers([end,end-1]);
                    end
                end
            else
                warning( 'Cannot move pooling before ReLU' );
            end
        elseif ismember( endAXType, 'ReLU' )
            nlIdx = alN;
        end
    end
    
    bnIdx = find( ismember( X(idx_skip).conv_combo_param.aux_layers(1:nlIdx-1), {'BN'} ), 1, 'last' );
    poolIdx = find( ismember( X(idx_skip).conv_combo_param.aux_layers(1:nlIdx-1), {'Pooling'} ), 1, 'last' );
    if isempty(bnIdx)
        X(idx_skip).conv_combo_param.aux_layers = [ ...
            X(idx_skip).conv_combo_param.aux_layers(1:nlIdx-1), ...
            'LadderSkip', ...
            X(idx_skip).conv_combo_param.aux_layers(nlIdx:end) ];
    else
        if isempty( poolIdx ) || poolIdx<bnIdx
            X(idx_skip).conv_combo_param.aux_layers = [ ...
                X(idx_skip).conv_combo_param.aux_layers(1:bnIdx-1), ...
                'BN1', 'LadderSkip', 'BN2', ...
                X(idx_skip).conv_combo_param.aux_layers(nlIdx+1:end) ];
        else
            X(idx_skip).conv_combo_param.aux_layers = [ ...
                X(idx_skip).conv_combo_param.aux_layers(1:bnIdx-1), ...
                'BN1', ...
                X(idx_skip).conv_combo_param.aux_layers(bnIdx+1:poolIdx), ...
                'LadderSkip', 'BN2', ...
                X(idx_skip).conv_combo_param.aux_layers(poolIdx+1:end) ];
        end
    end
    X(idx_skip) = caffeproto_convcombo_set_aux_blobs( X(idx_skip), ... 
        'LadderSkip', [], {ladderSkipName} );
else
    X(end+1).name = { ladderSkipName };
    X(end).type = { 'Split:Branch' };
    X(end).tag  = { 'LadderSkip' };
    X(end).bottom = subS(idx_ladder).bottom(2);
    X(end).top = {subS(idx_ladder).bottom{2}, ladderSkipName};
    X(end).replace_at = idx_skip;
end

% for top-down stream, put the combinator before non-linearity
assert( strcmp( X(idx_topdown).type{1}, 'ConvCombo' ), 'wrong layer type' );
alN = length( X(idx_topdown).conv_combo_param.aux_layers );
nlIdx = alN+1;
if alN>0
    endAXType = X(idx_topdown).conv_combo_param.aux_layers{alN};
    if ismember( endAXType, 'ReLU' )
       nlIdx = alN;
    end
end
X(idx_topdown).conv_combo_param.aux_layers = [ ...
    X(idx_topdown).conv_combo_param.aux_layers(1:nlIdx-1), ...
    'LadderCombined', ...
    X(idx_topdown).conv_combo_param.aux_layers(nlIdx:end) ];
X(idx_topdown) = caffeproto_convcombo_set_aux_blobs( X(idx_topdown), ... 
        'LadderCombined', {ladderSkipName}, [] );


return;
