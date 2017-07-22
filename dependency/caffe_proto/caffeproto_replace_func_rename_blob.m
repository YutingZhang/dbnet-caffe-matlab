function X = caffeproto_replace_func_rename_blob( varargin )


if strcmp( varargin{1}, 'extend' )
    X = {};
    return;
end

if strcmp( varargin{1}, 'adjacent' )
    X = [0];
    return;
end


ARGS = varargin{end};
srcBlobNames = ARGS{1};
dstBlobNames = ARGS{2};

if isempty(srcBlobNames), 
    X = [];
    return;
end

if ischar(srcBlobNames), srcBlobNames={srcBlobNames}; end
if ischar(dstBlobNames), dstBlobNames={dstBlobNames}; end

assert( numel(srcBlobNames) == numel(dstBlobNames), ...
    'src and dst should have the same size' );

subS = varargin{1};

X = subS;
CODE_BOOK = [vec(srcBlobNames),vec(dstBlobNames)];
[X.bottom, is_replaced_b] = map_with_codebook( subS.bottom, CODE_BOOK );
[X.top, is_replaced_t] = map_with_codebook( subS.top, CODE_BOOK );

if ~any(is_replaced_b) && ~any(is_replaced_t)
    X = [];
end
