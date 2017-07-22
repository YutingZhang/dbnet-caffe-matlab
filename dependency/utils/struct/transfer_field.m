function [DST,SRC] = transfer_field( DST, SRC, fieldName, todoRM )
% [DST,SRC] = transfer_field( SRC, DST, fieldName, todoRM )

if ~exist('todoRM','var') || isempty(todoRM)
    todoRM = 1;
end

assert( isstruct(SRC) && isstruct(DST), ...
    'first two arguents must both be struct' );
assert( isequal( size(SRC),size(numel(DST)) ), ...
    'first two arguments must have the same size' );

if iscell( fieldName )
    assert( length(fieldName)==2, 'wrong fieldName' );
    dstFieldName = fieldName{1}; % DST first
    srcFieldName = fieldName{2}; % SRC second
else
    srcFieldName = fieldName;
    dstFieldName = fieldName;
end

if isfield(SRC,srcFieldName)
    eval(['DST(:).' dstFieldName ' = SRC(:).' srcFieldName ';']);
    if todoRM, 
        SRC = rmfield(SRC, srcFieldName);
    else
        eval(['SRC(:).' srcFieldName ' = deal([]);']);
    end
else
    eval(['DST(:).' dstFieldName ' = deal([]);']);
end

