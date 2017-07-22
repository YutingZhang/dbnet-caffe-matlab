function A=cmd2struct(varargin)
% A=cmd2struct(CMD1,CMD2,...)
% CMD* can be either a string or a string cell

V = varargin;
cell_idx = cellfun(@iscell,V);
V(~cell_idx) = arrayfun(@(a) {a},V(~cell_idx));
V = [V{:}];

A = exec2struct(V,'inline');

end
