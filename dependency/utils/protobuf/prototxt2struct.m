function S = prototxt2struct( fn, input_type, cell4single_string, enable_file_cache )

if ~exist( 'input_type', 'var' ) || isempty(input_type)
    input_type = 'file';
end

if ~exist( 'cell4single_string', 'var' ) || isempty(cell4single_string)
    cell4single_string = 1;
end

if ~exist( 'enable_file_cache', 'var' ) || isempty(enable_file_cache)
    enable_file_cache = 1;
end

% load prototxt file
switch input_type
    case 'file'
        if enable_file_cache && ...
                exist('cached_file_func', 'file') == 2
            % use file cache if possible
            S = cached_file_func( ...
                @(a) prototxt2struct(a, 'file', cell4single_string, 0), ...
                fn, 'prototxt2struct', 200 );
            return;
        end
        T = readtextfile(fn);
    case 'string'
        T = fn;
    otherwise
        error( 'unrecognized input_type' );
end
T = [' ', T, ' '];  

% parse

spaceCh = sprintf(' \t\n\r');
enterCh = sprintf('\n\r');

bN = ismember(T,spaceCh); bN(end+1)=true;
bE = ismember(T,enterCh); bE(end+1)=true;

isExpectValue = 0;

S = struct();
nameStack = {};
curName   = [];

k = 0;
while k<length(T)
    k = k+1;
    if bN(k), continue; end
    if T(k)=='#'    % comment
        k=k+1;
        while ~bE(k), k=k+1; end
        continue;
    end
    if isExpectValue
        if isempty(nameStack)
            curStructPath = 'S';
        else
            curStructPath = cellfun( @(a) {sprintf('.%s(end)',a)}, nameStack );
            curStructPath = ['S' curStructPath{:}];
        end
        curFieldInitialized = isfield( eval(curStructPath), curName );
        curFieldCount = 0;
        curFieldVal = [];
        if curFieldInitialized
            curFieldVal = eval(sprintf('%s.%s', curStructPath, curName));
            if ischar(curFieldVal), curFieldCount = 1;
            else curFieldCount = numel(curFieldVal); end
            curFieldInitialized = (curFieldCount>0);
        end
        if T(k)=='{'
            if curFieldInitialized
                fdNames = fieldnames( eval(sprintf('%s.%s', curStructPath, curName)) );
                if isempty(fdNames)
                    assignhere( sprintf('%s.%s(end+1)', curStructPath, curName), struct() );
                else
                    assignhere( sprintf('%s.%s(end+1).%s', curStructPath, curName, fdNames{1}), [] );
                end
            else
                assignhere( sprintf('%s.%s', curStructPath, curName), struct() );
            end
            nameStack = [nameStack; {curName}];
        else
            is_string = 0;
            if T(k)=='"'    % string
                k0 = k; k=k+1;
                while T(k)~='"', k=k+1; end
                v = { T(k0+1:k-1) }; % include ""
                is_string = 1;
            elseif T(k)==''''    % string
                k0 = k; k=k+1;
                while T(k)~='''', k=k+1; end
                v = { T(k0+1:k-1) }; % include ""
            else
                k0 = k; k=k+1;
                while ~bN(k) && T(k)~='}', k=k+1; end
                val_str = T(k0:k-1);
                v = str2num(val_str);   % try double first
                if isempty(v) % enum
                    v = proto_enum_class(val_str);
                end
                if T(k) == '}', k=k-1; end
            end
            if curFieldInitialized
                if is_string && ~cell4single_string && curFieldCount==1
                    assignhere( sprintf('%s.%s', curStructPath, curName), {curFieldVal} );
                    assignhere( sprintf('%s.%s(end+1)', curStructPath, curName), v );
                else
                    assignhere( sprintf('%s.%s(end+1)', curStructPath, curName), v );
                end
            else
                if is_string && ~cell4single_string
                    v = v{1};
                end
                assignhere( sprintf('%s.%s', curStructPath, curName), v );
            end
        end
        isExpectValue = ~isExpectValue;
    else
        if T(k)=='}'   % end of struct
            nameStack(end) = [];
        else   % name
            k0 = k; k=k+1;
            while T(k)~=':' && T(k)~='{', k=k+1; end
            curName = strtrim(T(k0:k-1));
            if T(k)=='{', 
                k=k-1; 
            end
            isExpectValue = ~isExpectValue;
        end
    end
end
