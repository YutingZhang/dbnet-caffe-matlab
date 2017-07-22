function varargout = display_struct( S, max_depth, serialize_func, var_name )
% display_struct( S, [max_depth], [serialize_func] )
% output_str = display_struct(...);

if ~exist( 'max_depth', 'var' ) || isempty(max_depth)
    max_depth = inf;
end

if ~exist( 'serialize_func', 'var' ) || isempty(serialize_func)
    serialize_func = @any2str;
end

if ~exist( 'var_name', 'var' ) || isempty(var_name)
    var_name = inputname(1);
    if isempty(var_name)
        var_name = '~';
    end
end

if max_depth<=0 || ~isstruct(S) || isempty(S) || isempty(fieldnames(S))
    output_str = serialize_func( S );
else
    sub_struct_display_func = @(varargin) display_struct( ...
        varargin{:}, max_depth-1, serialize_func, [var_name, '.', inputname(1)] );
    szS = size(S);
    ndS = ndims(S);
    curSub  = cell(1,ndS);
    szS_str = vec2str(szS);
    output_str_cell = cell(2,numel(S));
    for k = 1:numel(S)
        [curSub{:}] = ind2sub( szS, k );
        output_str_cell{1,k} = sprintf( '%s: (%s)/(%s)\n', ...
            var_name, vec2str(cell2mat(curSub)), szS_str );
        field_str = display_grouped_vars( ...
            sub_struct_display_func, S(k), []);
        field_str_cell = strsplit( field_str, sprintf('\n') );
        field_str_cell(2,:) = {'  '};
        field_str_cell = field_str_cell([2 1],:);
        field_str_cell(3,1:(end-1)) = {sprintf('\n')};
        output_str_cell{2,k} = cat(2, field_str_cell{:});
    end
    output_str_cell(3,1:end-1) = {sprintf('\n')};
    output_str = cat(2, '', output_str_cell{:});
end

if nargout==0
    fprintf(1, '%s\n', output_str);
else
    varargout{1} = output_str;
end
