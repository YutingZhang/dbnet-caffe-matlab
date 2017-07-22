function [X, varargout] = caffeproto_replace_func_clear_aux(varargin)

if strcmp(varargin{1},'extend')
    X = {};
    return;
elseif strcmp(varargin{1},'adjacent')
    X = [0];
    return;
end

subS = varargin{1};

X = subS;
isModified = 0;

if isfield(subS, 'aux')
    isModified = 1;
    X = rmfield( X, 'aux' );
end

if isfield(subS, 'nonprefixable_bottom')
    isModified = 1;
    X = rmfield( X, 'nonprefixable_bottom' );
end

if isfield(subS, 'conv_aux')
    isModified = 1;
    X = rmfield( X, 'conv_aux' );
end

if ~isModified, X = []; end
