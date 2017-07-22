function T = cell2csv( C )

B = cellfun( @(c) {any2text4csv(c)}, C );
B = reshape( B.', 1, size(B,2), size(B,1) );
B(2,1:end-1,:)   = {','};
B(2,end,:)       = {sprintf('\n')};

T = [B{:}];

end

function t = any2text4csv(c)

if isnumeric(c)
    t = num2str(c,'%g');
else
    if ischar(c)
        t = c;
    else
        t = any2str(c);
    end
    if isnan(str2double(t))
        t = strrep( t, '"', '""' );
        t = [ '"' t '"' ];
    end
end

end
