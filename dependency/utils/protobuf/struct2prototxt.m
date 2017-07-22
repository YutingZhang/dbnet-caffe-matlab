function varargout = struct2prototxt( S, fn )

if ischar(S)
    S = evalin('caller','S');
end

T = struct2prototxt_rec( S );

if exist('fn', 'var')
    fid = fopen(fn,'w');
    fprintf( fid, '%s', T );
    fclose(fid);
    if nargout>=1
        varargout{1} = T;
    end
else
    varargout{1} = T;
end

function [T, Tsh ] = struct2prototxt_rec( S )

tabStr = '  ';
tabSize = 8;
maxInline = 40;
enterCh = sprintf('\n');
ctCh    = sprintf('$$GAP$$');

T = {''};
Tsh = {''};

validFIdxb    = ~structfun(@isempty, S);
substructIdxb = structfun(@isstruct, S);
F = fieldnames(S);
F = F(validFIdxb);
substructIdxb = substructIdxb(validFIdxb);

if any(validFIdxb)
    vls = cellfun(@length,F);
    vts = ceil(vls/tabSize);
    vls(substructIdxb) = vls(substructIdxb) - 1;
    vlref = accumarray(vts,vls,[],@max);
end

for k=1:numel(F)
    E = eval(sprintf('S.%s',F{k}));
    if ischar(E), E = {E}; end
    if isempty(E), continue; end
    for j = 1:numel(E)
        t = E(j);
        R = [];
        if isstruct(t)
            [R, Rsh] = struct2prototxt_rec( t );
            %if isempty(strtrim(Rsh))
            %    R = ''; %??
            %else
            if length(Rsh) <= maxInline
                R = Rsh;
                R(R==enterCh) = ' ';
                R = [ctCh ' { ', R,'}'];
            else
                R = strsplit(R,enterCh);
                R(cellfun(@isempty,R)) = [];
                R = cellfun( @(a) { [tabStr a enterCh] }, R );
                R = [' {' enterCh R{:} '}'];
            end
        elseif islogical(t)
            if t, R = [':' ctCh 'true'];
            else  R = [':' ctCh 'false']; end
        elseif isnumeric(t)
            R = [':' ctCh sprintf( '%.12g', t )];
        elseif iscell(t) && isscalar(t) && ischar(t{1})
            if any(t{1}=='"')
                R = [':' ctCh sprintf( '''%s''', t{1} )];
            else
                R = [':' ctCh sprintf( '"%s"', t{1} )];
            end
        elseif isa(t,'proto_enum_class')
            R = [':' ctCh sprintf( '%s', t.val() )];
        else
            error('failed to parse');
        end
        if ~isempty(R)
            vl = length(F{k});
            Tsh{end+1} = [F{k} strrep(R,ctCh, ' ') enterCh];
            R=strrep(R, ctCh, repmat(' ', 1, vlref(ceil(vl/tabSize))-vl+1) );
            T{end+1} = [F{k} R enterCh];
        end
    end
end

T = [T{:}];
Tsh = [Tsh{:}];


