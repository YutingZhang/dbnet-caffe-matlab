function varargout = display_grouped_vars( ...
    serialize_func, var_struct, group_titles, varargin)
% display_grouped_vars(SERIALIZE_FUNC, VAR_STRUCT, GROUP_TITLES, VAR_GROUP1, VAR_GROUP2, ...)
% output_str = display_grouped_vars( ... )

if isempty( serialize_func )
    serialize_func = @any2str;
end

if ~exist('group_titles', 'var') || isempty( group_titles )
    group_titles = {};
end

vargroups = vec(varargin).';
for k = (length(group_titles)+1):length(vargroups)
    group_titles(k) = {['*group ' int2str(k)]};
end
if isempty(group_titles)
    group_titles = {''};
else
    group_titles{end+1} = '*unknown';
    for k = 1:numel(group_titles)
        group_titles{k} = [group_titles{k}, ', '];
    end
end

if isempty(vargroups)
    G = zeros(1,0);
    V = cell(1,0);
else
    G = cell2mat( arrayfun( @(i,n) {repmat(i,1,n)}, ...
        1:length(vargroups), cellfun( @numel, vargroups ) ) );
    V = cellfun( @(a) {reshape(a,1,numel(a))}, vargroups );
    V = cat(2,V{:});
end

var_names = fieldnames( var_struct );
[known_idxb,I] = ismember( var_names , V );
J = zeros(size(I));
J(~known_idxb) = length(group_titles);
J(known_idxb)  = G(I(known_idxb)); 

first_tab_str = [repmat(' ',1,3), ' '];
tab_str = repmat(' ', size(first_tab_str) );

output_str_cell = cell(1,length(var_names));
for k = 1:length(var_names)
    ss = substruct('.',var_names{k});
    %v_str = feval( serialize_func, subsref(var_struct,ss) );
    v_str = feval_with_var_name( serialize_func, var_names{k}, subsref(var_struct,ss) );
    v_str_cell = strsplit( v_str, sprintf('\n') );
    v_str_cell(2,:) = {tab_str};
    v_str_cell(2,1) = {first_tab_str};
    v_str_cell = v_str_cell([2 1],:);
    v_str_cell(3,1:(end-1)) = {sprintf('\n')};
    v_str = cat(2,v_str_cell{:});
    output_str_cell{k} = sprintf( '%s = \t[%s%s]\n%s\n' , var_names{k}, ...
        group_titles{ J(k) }, ...
        class(subsref(var_struct,ss)), ...
        v_str );
end

output_str = cat(2, '', output_str_cell{:});

if ~isempty(output_str) && output_str(end) == sprintf('\n')
    output_str(end) = [];
end

if nargout==0
    fprintf(1, '%s\n', output_str);
else
    varargout{1} = output_str;
end

function output___ = feval_with_var_name( func___, var_name___, value___ )

assignhere( var_name___, value___ );
output___ = eval( sprintf('func___(%s)', var_name___) );

