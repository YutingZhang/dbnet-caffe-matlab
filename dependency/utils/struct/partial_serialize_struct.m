function T = partial_serialize_struct( S )

F = fieldnames(S);
T = cell(1,length(F));
for i = 1:length(F)
    v = eval( ['S.',F{i}] );
    if isnumeric( v ) || islogical(v)
        if numel(v) == 0
            t = '[empty]';
        elseif numel(v)<=4
            t = sprintf('%g,', v);
            t = t(1:end-1);
        else
            v=v(1:4);
            t = sprintf('%g,', v);
            t = [t,'...'];
        end
    elseif ischar(v)
        if length(v)<30
            t = v;
        else
            t = [v(1:27), '...'];
        end
    elseif isstruct(v)
        t = '';
        for k = 1:length(v)
            if k>1, t=[t ',']; end
            t = [ t '{' partial_serialize_struct( v(k) ) '}' ];
        end
    else
        t = sprintf('[%s]',class(v));
    end
    t(ismember(t,'/\:?*"<>')) = '_'; % potential illege characters in file names
    T{i} = [F{i},'=',t,';'];
end
T = cell2mat(T);
if ~isempty(T)
    T = T(1:end-1);
end

end
