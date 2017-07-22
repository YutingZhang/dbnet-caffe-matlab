function T = readtextfile(fn,mode_str)

if ~exist('mode_str','var') || isempty(mode_str)
    mode_str = 'text';
end

assert( ismember(mode_str, {'text','lines'}), 'Unrecognized mode' );

% fid = fopen(fn,'r');
% T = {};
% tline = fgetl(fid);
% while ischar(tline)
%     T{end+1} = [tline sprintf('\n')];
%     tline = fgetl(fid);
% end
% fclose(fid);
%
% switch mode_str
%     case 'lines'
%         T = reshape(T,numel(T),1);
%     case 'text'
%         T = cat(2, T{:});
%     otherwise
%         error( 'Unrecognized mode' );
% end

fid = fopen(fn,'rb');
fseek(fid,0,'eof');
fileLen = ftell(fid);
fseek(fid,0,'bof');
T = fread(fid,fileLen,'uint8=>char').';
fclose(fid);

T = strrep(T,sprintf('\r\n'),sprintf('\n'));
T = strrep(T,sprintf('\n\r'),sprintf('\n'));
T = strrep(T,sprintf('\r'),sprintf('\n'));

switch mode_str
    case 'lines'
        T = strsplit( T, sprintf('\n') ).';
    case 'text'
    otherwise
        error( 'Unrecognized mode' );
end

