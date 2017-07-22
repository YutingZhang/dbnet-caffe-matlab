function T = readlines( file_name, ignore_empty )

if ~exist('ignore_empty','var') || isempty(ignore_empty)
    ignore_empty = 0;
end


T = {};

fid = fopen(file_name,'r');
tline = fgetl(fid);
while ischar(tline)
    T = [T;{tline}];
    tline = fgetl(fid);
end

% fseek( fid, 0, 'eof' );
% siz = ftell(fid);
% fseek( fid, 0, 'bof' );
% T = fread(fid,siz,'char=>char');
% 

% T = reshape( T, 1, numel(T) );
% T = strrep( T, sprintf('\r\n'), sprintf('\n') );
% T = strrep( T, sprintf('\r'), sprintf('\n') );
% if ~isempty(T) && T(end) == sprintf('\n'), 
%     T=T(1:end-1); 
% end
% T = strsplit( T, sprintf('\n') ).';

fclose(fid);

if ignore_empty
    T( cellfun(@(a) isempty(strtrim(a)), T) ) = [];
end

