function T = any2str( A )

if isempty(A)
    T = sprintf('[empty %s]', class(A));
elseif ischar(A)
    T = A;
elseif isstruct(A) && isempty( fieldnames(A) )
    T = sprintf( '%dx', size(A) );
    T = T(1:end-1);
    T = [T ' struct with no field'];
else
    T = evalc( 'disp(A)' );
    if ~isempty(T)
        R = evalc( 'disp(0)' );
        while R(end)==sprintf('\n') ...
            && ~isempty(T) && T(end)==sprintf('\n')
                T = T(1:end-1);
                R = R(1:end-1);
        end        
    end
    if  ~isempty(T)
        T = strsplit(T, sprintf('\n'));
        preS = zeros( size(T) );
        for k = 1:numel(preS)
            psk = find(T{k}~=' ',1);
            if isempty(psk)
                preS(k) = length(T{k});
            else
                preS(k) = psk-1;
            end
        end
        min_preS = min(preS);
        T(1:end-1) = cellfun( @(a) {[a(min_preS+1:end) sprintf('\n')]}, T(1:end-1));
        T{end} = T{end}(min_preS+1:end);
        T = [T{:}];
    end
end

    
end
