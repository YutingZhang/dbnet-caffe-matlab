function [X, varargout] = caffeproto_replace_func_cae_absorb( varargin )

if ischar(varargin{1})
    if strcmp(varargin{1},'extend')
        X = { 'iterative', @(varargin) caffeproto_replace_func_cae_absorb( 'comb-with-pre', varargin{:} ), ...
            };
        return;
    end
end

VAR_IN = varargin(2:end);

assert( strcmp( varargin{1}, 'comb-with-pre' ), 'wrong internal state' );

if ischar( VAR_IN{1} )
    if strcmp( VAR_IN{1}, 'extend' )
        X = {};
        return
    elseif strcmp( VAR_IN{1}, 'adjacent' )
        X = [0 1; 0 0];
        varargout{1} = [1]; % in
        varargout{2} = [2]; % out
        return;
    end
end

subS = VAR_IN{1};
ARGS = VAR_IN{end};

if strcmp( subS(2).type, 'CAEStack' )
%     is_matched = ...
%         ( strcmp( subS(1).type, 'ConvCombo' ) || ...
%         ( strcmp( subS(1).type, 'Combo' ) || strcmp( subS(1).name, '[CAE-MID]' ) ) );
%     if is_matched
        X = subS(2);
        X.bottom = subS(1).bottom;
        X.layers = cat_struct( 2, X.layers, subS(1) );
        X.replace_at = 2;
%     else
%         X = [];
%     end
else
    X = [];
end

