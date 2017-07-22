function S1 = partial_struct( S0, varargin )
% S1 = partial_struct( S0, REGEXP1, REGEXP2, ... )
% S1 = partial_struct( S0, '@include', REGEXP1, REGEXP2, ... )
% S1 = partial_struct( S0, '@exclude', REGEXP1, REGEXP2, ... )
% Remark REGEXPi are comined by OR

if isempty(varargin)
    S1 = repmat( struct(), size(S0) );
    return;
end

VAR_IN = varargin;

is_exclusion = 0;
if strcmp( VAR_IN{1}, '@exclude' )
    is_exclusion = 1;
    VAR_IN = VAR_IN(2:end);
elseif strcmp( VAR_IN{1}, '@include' )
    is_exclusion = 0;
    VAR_IN = VAR_IN(2:end);
end

chosenFiledNames = VAR_IN;

cellIdx = find(cellfun( @iscell,chosenFiledNames ),1);
while ~isempty(cellIdx)
    chosenFiledNames = [chosenFiledNames(1:cellIdx-1), ...
        reshape( chosenFiledNames{cellIdx}, 1, numel(chosenFiledNames{cellIdx}) ), ... 
        chosenFiledNames(cellIdx+1:end)];
    cellIdx = find(cellfun( @iscell, chosenFiledNames ),1);
end

chosenFiledNames = unique(chosenFiledNames); 

try
    VA = chosenFiledNames;
    VA(2,:) = {[]};
    struct(VA{:});
    use_regexp = 0;
catch
    use_regexp = 1;
end


n = length(chosenFiledNames);

F = fieldnames(S0);
C = false( length(F), n );
if use_regexp
    for k = 1:n
        C(:,k) = cellfun( @(a) ~isempty(regexp(a,['^' chosenFiledNames{k} '$'],'once')), F );
    end
    Cf = any(C,2);
else
    Cf = ismember( F, chosenFiledNames );
end

if is_exclusion
    Cf = ~Cf;
end

CMD = cellfun( @(a) {sprintf(' ''%s'', {S0.%s},', a, a)} , F(Cf) );
if isempty(CMD)
    S1 = repmat( struct(), size(S0) );
else
    CMD = [ CMD{:} ];
    CMD = sprintf( 'struct(%s )', CMD(1:end-1) );
    S1 = eval( CMD );
    S1 = reshape(S1,size(S0));
end


end
