function block_input(blk_duration)

if ~exist('blk_duration','var') || isempty(blk_duration)
    blk_duration = inf;
end

fprintf( 'DEBUG: Block all the input for %gs to prevent pasted info\n', blk_duration );
t = tic;
tmp_input = ' ';
while toc(t)<blk_duration
    t = tic;
    if isempty(tmp_input)
        tmp_input = input('','s');
    else
        tmp_input = input('B>>','s');
    end
end
fprintf( 'DEBUG: Released from blocking\n' );

end
