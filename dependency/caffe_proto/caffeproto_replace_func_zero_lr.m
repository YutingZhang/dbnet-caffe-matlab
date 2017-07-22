function [X,varargout] =caffeproto_replace_func_zero_lr( varargin )

if strcmp(varargin{1},'extend')
    X = {};
    return;
elseif strcmp(varargin{1},'adjacent')
    X = [0];
    return;
end

subS = varargin{1};

%
Pdef = struct();
Pdef.mode = 'instant'; % 'instant', 'postpone', 'apply'

% use general param first
PARAM = scalar_struct(varargin{end}{:});
PARAM = xmerge_struct('always','always', Pdef, PARAM);


X = [];
if isfield(subS,'param') && isfield(subS.param,'lr_mult')
    X = subS;
    for k = 1:length(X.param)
        switch PARAM.mode
            case 'instant'
                X.param(k).lr_mult = 0;
            case 'postpone'
                X.param(k).lr_mult_zero = 1;
            case 'apply'
                if try_to_eval( 'X.param(k).lr_mult_zero', 0 )
                    X.param(k).lr_mult = 0;
                end
                X.param(k).lr_mult_zero = [];
            otherwise
                error( 'Unrecognized zero_lr mode' );
        end
    end
end
