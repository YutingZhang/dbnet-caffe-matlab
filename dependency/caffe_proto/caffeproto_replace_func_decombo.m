function X = caffeproto_replace_func_decombo( varargin )

if ischar(varargin{1})
    if strcmp( varargin{1}, 'adjacent' )
        X = [0];
    elseif strcmp( varargin{1}, 'extend' )
        X = {};
    else
        error( 'Unrecognized mode' );
    end
    return;
end

subS = varargin{1};
if strcmp( subS(1).type,'Combo' )
    ARGS = varargin{end};
    if isempty(ARGS) || strcmp(subS.name{1}, ARGS{1})
        X = subS.layers;
        % update bottom and top blob names if changes
        % bottom
        if ~isempty(subS.bottom)
            b = ~cellfun(@strcmp, subS.bottom, subS.bottom0);
            newNames = subS.bottom(b); oldNames = subS.bottom0(b);
            if ~isempty(newNames)
                for k = 1:length(X)
                    if ~isempty(X(k).bottom)
                        [is_matched, matched_idx] = ismember( X(k).bottom, oldNames );
                        X(k).bottom(is_matched) = newNames(matched_idx(is_matched));
                    end
                end
            end
        end
        % top
        if ~isempty(subS.top)
            b = ~cellfun(@strcmp, subS.top, subS.top0);
            newNames = subS.top(b); oldNames = subS.top0(b);
            if ~isempty(newNames)
                for k = 1:length(X)
                    if ~isempty(X(k).top)
                        [is_matched, matched_idx] = ismember( X(k).top, oldNames );
                        X(k).top(is_matched) = newNames(matched_idx(is_matched));
                    end
                end
            end
        end        
    else
        X = [];
    end
else
    X = [];
end
